package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	cdplog "github.com/chromedp/cdproto/log"
	"github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
)

func TestParseDevice(t *testing.T) {
	for name, expected := range map[string]mb.Device{"auto": mb.DeviceAuto, "cpu": mb.DeviceCPU, "vulkan": mb.DeviceVulkan, "CPU": mb.DeviceCPU} {
		actual, err := parseDevice(name)
		if err != nil || actual != expected {
			t.Fatalf("parseDevice(%q)=(%d,%v), want %d", name, actual, err, expected)
		}
	}
	if _, err := parseDevice("cuda"); err == nil {
		t.Fatal("unknown device was accepted")
	}
}

func TestNativeHTTPFlow(t *testing.T) {
	libraryPath, modelPath, stylesPath := os.Getenv("MOTIONBRICKS_LIB"), os.Getenv("MOTIONBRICKS_MODEL"), os.Getenv("MOTIONBRICKS_STYLES")
	if libraryPath == "" || modelPath == "" || stylesPath == "" {
		t.Skip("native demo paths are not configured")
	}
	demo, err := loadDemoServer(libraryPath, modelPath, stylesPath, mb.DeviceCPU)
	if err != nil {
		t.Fatal(err)
	}
	defer demo.Close()
	server := httptest.NewServer(demo.routes())
	defer server.Close()
	response, err := http.Get(server.URL + "/api/meta")
	if err != nil {
		t.Fatal(err)
	}
	var metadata struct {
		FPS    int         `json:"fps"`
		Joints []mb.Joint  `json:"joints"`
		Styles []styleInfo `json:"styles"`
	}
	if err = json.NewDecoder(response.Body).Decode(&metadata); err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusOK || metadata.FPS != 30 || len(metadata.Joints) != 34 || len(metadata.Styles) < 10 {
		t.Fatalf("invalid metadata: status=%d fps=%d joints=%d styles=%d", response.StatusCode, metadata.FPS, len(metadata.Joints), len(metadata.Styles))
	}
	post := func(path string, value any, output any) {
		t.Helper()
		body, _ := json.Marshal(value)
		reply, postErr := http.Post(server.URL+path, "application/json", bytes.NewReader(body))
		if postErr != nil {
			t.Fatal(postErr)
		}
		defer reply.Body.Close()
		if reply.StatusCode != http.StatusOK {
			var failure any
			_ = json.NewDecoder(reply.Body).Decode(&failure)
			t.Fatalf("%s: status=%d body=%v", path, reply.StatusCode, failure)
		}
		if err := json.NewDecoder(reply.Body).Decode(output); err != nil {
			t.Fatal(err)
		}
	}
	var initial planResponse
	post("/api/session", sessionRequest{Style: "walk"}, &initial)
	if initial.Session == "" || initial.Motion == nil || initial.Motion.Joints != 34 || initial.Motion.Frames < 24 ||
		initial.Targets == nil || initial.Targets.Frames != 4 || initial.Targets.Joints != 34 ||
		len(initial.Targets.Roots) != 12 || len(initial.Targets.Rotations) != 4*34*4 {
		t.Fatalf("invalid initial response: %+v", initial)
	}
	turnedStyle := ""
	for _, style := range metadata.Styles {
		if style.Name == "walk_zombie" {
			turnedStyle = style.Name
			break
		}
		if style.Name != "walk" && turnedStyle == "" {
			turnedStyle = style.Name
		}
	}
	if turnedStyle == "" {
		t.Fatal("no alternate upstream style is available")
	}
	var turned planResponse
	post("/api/plan", planRequest{Session: initial.Session, Style: turnedStyle, Move: [2]float32{1, 0}, Facing: [2]float32{1, 0}, Advance: 3, Seed: 77}, &turned)
	if turned.Style != turnedStyle || turned.Motion == nil || turned.Motion.Joints != 34 || len(turned.Motion.Rotations) != int(turned.Motion.Frames*34*4) ||
		turned.Targets == nil || turned.Targets.Frames != 4 || len(turned.Targets.Rotations) != 4*34*4 {
		t.Fatalf("invalid turned response: style=%q motion=%+v", turned.Style, turned.Motion)
	}
}

func TestHeadlessChrome(t *testing.T) {
	libraryPath, modelPath, stylesPath := os.Getenv("MOTIONBRICKS_LIB"), os.Getenv("MOTIONBRICKS_MODEL"), os.Getenv("MOTIONBRICKS_STYLES")
	if libraryPath == "" || modelPath == "" || stylesPath == "" {
		t.Skip("native demo paths are not configured")
	}
	chrome := os.Getenv("MOTIONBRICKS_CHROME")
	if chrome == "" {
		var err error
		chrome, err = exec.LookPath("chromium")
		if err != nil {
			t.Skip("chromium is not installed")
		}
	}
	demo, err := loadDemoServer(libraryPath, modelPath, stylesPath, mb.DeviceCPU)
	if err != nil {
		t.Fatal(err)
	}
	defer demo.Close()
	server := httptest.NewServer(demo.routes())
	defer server.Close()

	options := append([]chromedp.ExecAllocatorOption{}, chromedp.DefaultExecAllocatorOptions[:]...)
	options = append(options,
		chromedp.ExecPath(chrome),
		chromedp.Flag("no-sandbox", true),
		chromedp.Flag("disable-dev-shm-usage", true),
		chromedp.Flag("use-angle", "swiftshader"),
		chromedp.Flag("enable-unsafe-swiftshader", true),
		chromedp.WindowSize(1280, 800),
	)
	allocator, cancelAllocator := chromedp.NewExecAllocator(context.Background(), options...)
	defer cancelAllocator()
	browser, cancelBrowser := chromedp.NewContext(allocator)
	defer cancelBrowser()
	chromedp.ListenTarget(browser, func(event any) {
		switch event := event.(type) {
		case *cdplog.EventEntryAdded:
			t.Logf("browser log: %s", event.Entry.Text)
		case *runtime.EventConsoleAPICalled:
			t.Logf("browser console: %v", event.Args)
		case *runtime.EventExceptionThrown:
			t.Logf("browser exception: %s", event.ExceptionDetails.Text)
		}
	})
	ctx, cancel := context.WithTimeout(browser, 30*time.Second)
	defer cancel()

	waitFor := func(expression, description string) chromedp.Action {
		return chromedp.ActionFunc(func(ctx context.Context) error {
			deadline := time.Now().Add(25 * time.Second)
			for time.Now().Before(deadline) {
				var ready bool
				if evaluateErr := chromedp.Evaluate(expression, &ready).Do(ctx); evaluateErr != nil {
					return evaluateErr
				}
				if ready {
					return nil
				}
				time.Sleep(100 * time.Millisecond)
			}
			var diagnostic any
			_ = chromedp.Evaluate(`({status: document.documentElement.dataset.testStatus, sequence: document.documentElement.dataset.planSequence, moveX: document.documentElement.dataset.plannedMoveX, moveZ: document.documentElement.dataset.plannedMoveZ, scripts: [...document.scripts].map(s => ({src:s.src,type:s.type})), resources: performance.getEntriesByType("resource").map(r => r.name)})`, &diagnostic).Do(ctx)
			return fmt.Errorf("%w for %s: %#v", errors.New("timeout waiting for browser"), description, diagnostic)
		})
	}
	var initialScreenshot, movingScreenshot, screenshot []byte
	var message string
	err = chromedp.Run(ctx,
		chromedp.ActionFunc(func(ctx context.Context) error {
			if enableErr := cdplog.Enable().Do(ctx); enableErr != nil {
				return enableErr
			}
			return runtime.Enable().Do(ctx)
		}),
		chromedp.Navigate(server.URL+"/"),
		waitFor(`document.documentElement.dataset.testStatus === "ready"`, "initial plan"),
		chromedp.FullScreenshot(&initialScreenshot, 90),
		chromedp.Click(`.pad button[data-key="w"]`, chromedp.ByQuery),
		waitFor(`Number(document.documentElement.dataset.planSequence) >= 2 && document.documentElement.dataset.plannedMoveZ === "1"`, "forward pad-button plan"),
		chromedp.FullScreenshot(&movingScreenshot, 90),
		chromedp.Click(`.pad button[data-key="w"]`, chromedp.ByQuery),
		waitFor(`Number(document.documentElement.dataset.planSequence) >= 3 && document.documentElement.dataset.plannedMoveX === "0" && document.documentElement.dataset.plannedMoveZ === "0"`, "pad-button stop plan"),
		chromedp.Evaluate(`dispatchEvent(new KeyboardEvent("keydown", {key:"d", bubbles:true}))`, nil),
		waitFor(`Number(document.documentElement.dataset.planSequence) >= 4 && document.documentElement.dataset.plannedMoveX === "1"`, "keyboard-right plan"),
		chromedp.Evaluate(`dispatchEvent(new KeyboardEvent("keyup", {key:"d", bubbles:true}))`, nil),
		waitFor(`Number(document.documentElement.dataset.planSequence) >= 5 && document.documentElement.dataset.plannedMoveX === "0" && document.documentElement.dataset.plannedMoveZ === "0"`, "keyboard stop plan"),
		chromedp.Navigate(server.URL+"/?test=1"),
		waitFor(`document.documentElement.dataset.testStatus === "passed" || document.documentElement.dataset.testStatus === "failed"`, "style-and-turn self-test"),
		chromedp.Text("#test-result", &message, chromedp.ByQuery),
		chromedp.FullScreenshot(&screenshot, 90),
	)
	if err != nil {
		t.Fatal(err)
	}
	var status string
	if err = chromedp.Run(ctx, chromedp.Evaluate(`document.documentElement.dataset.testStatus`, &status)); err != nil {
		t.Fatal(err)
	}
	if status != "passed" {
		t.Fatalf("browser self-test status=%q: %s", status, message)
	}
	if len(initialScreenshot) < 10_000 || len(movingScreenshot) < 10_000 || len(screenshot) < 10_000 {
		t.Fatalf("rendered screenshots are unexpectedly small: initial=%d moving=%d final=%d", len(initialScreenshot), len(movingScreenshot), len(screenshot))
	}
	artifact := os.Getenv("MOTIONBRICKS_SCREENSHOT")
	if artifact == "" {
		artifact = filepath.Join(t.TempDir(), "motionbricks-demo.png")
	}
	extension := filepath.Ext(artifact)
	if extension == "" {
		extension = ".png"
	}
	base := strings.TrimSuffix(artifact, filepath.Ext(artifact))
	artifacts := []struct {
		path string
		data []byte
	}{
		{path: base + "-initial" + extension, data: initialScreenshot},
		{path: base + "-moving" + extension, data: movingScreenshot},
		{path: artifact, data: screenshot},
	}
	for _, item := range artifacts {
		if err = os.WriteFile(item.path, item.data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("%s; screenshots: %s, %s, %s (final %s)", message, artifacts[0].path, artifacts[1].path, artifacts[2].path, fmt.Sprintf("%d bytes", len(screenshot)))
}
