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
	if initial.Session == "" || initial.Motion == nil || initial.Motion.Joints != 34 || initial.Motion.Frames < 24 {
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
	if turned.Style != turnedStyle || turned.Motion == nil || turned.Motion.Joints != 34 || len(turned.Motion.Rotations) != int(turned.Motion.Frames*34*4) {
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

	var screenshot []byte
	var message string
	err = chromedp.Run(ctx,
		chromedp.ActionFunc(func(ctx context.Context) error {
			if enableErr := cdplog.Enable().Do(ctx); enableErr != nil {
				return enableErr
			}
			return runtime.Enable().Do(ctx)
		}),
		chromedp.Navigate(server.URL+"/?test=1"),
		chromedp.ActionFunc(func(ctx context.Context) error {
			deadline := time.Now().Add(25 * time.Second)
			for time.Now().Before(deadline) {
				var status string
				if evaluateErr := chromedp.Evaluate(`document.documentElement.dataset.testStatus`, &status).Do(ctx); evaluateErr != nil {
					return evaluateErr
				}
				if status == "passed" || status == "failed" {
					return nil
				}
				time.Sleep(100 * time.Millisecond)
			}
			var diagnostic any
			_ = chromedp.Evaluate(`({status: document.documentElement.dataset.testStatus, scripts: [...document.scripts].map(s => ({src:s.src,type:s.type})), resources: performance.getEntriesByType("resource").map(r => r.name)})`, &diagnostic).Do(ctx)
			return fmt.Errorf("%w: %#v", errors.New("timeout waiting for browser self-test"), diagnostic)
		}),
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
	if len(screenshot) < 10_000 {
		t.Fatalf("rendered screenshot is unexpectedly small: %d bytes", len(screenshot))
	}
	artifact := os.Getenv("MOTIONBRICKS_SCREENSHOT")
	if artifact == "" {
		artifact = filepath.Join(t.TempDir(), "motionbricks-demo.png")
	}
	if err = os.WriteFile(artifact, screenshot, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Logf("%s; rendered screenshot: %s (%s)", message, artifact, fmt.Sprintf("%d bytes", len(screenshot)))
}
