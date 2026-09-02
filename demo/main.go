package main

import (
	"crypto/rand"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	mb "github.com/localai/motion-bricks.cpp/bindings/go"
)

//go:embed web/* web/vendor/*
var webFiles embed.FS

type styleInfo struct {
	Name  string  `json:"name"`
	Speed float32 `json:"speed"`
}

type session struct {
	mu      sync.Mutex
	agent   *mb.Agent
	planned bool
}

type demoServer struct {
	library  *mb.Library
	model    *mb.Model
	styles   map[string]*mb.Style
	ordered  []styleInfo
	joints   []mb.Joint
	planMu   sync.Mutex
	mu       sync.Mutex
	sessions map[string]*session
	static   http.Handler
}

type sessionRequest struct {
	Style string `json:"style"`
}
type planRequest struct {
	Session string     `json:"session"`
	Style   string     `json:"style"`
	Move    [2]float32 `json:"move"`
	Facing  [2]float32 `json:"facing"`
	Speed   *float32   `json:"speed,omitempty"`
	Seed    uint64     `json:"seed"`
	Advance uint32     `json:"advance"`
}
type planResponse struct {
	Session string        `json:"session"`
	Style   string        `json:"style"`
	Motion  *mb.Motion    `json:"motion"`
	Targets *mb.Keyframes `json:"targets"`
}

func parseDevice(value string) (mb.Device, error) {
	switch strings.ToLower(value) {
	case "auto":
		return mb.DeviceAuto, nil
	case "cpu":
		return mb.DeviceCPU, nil
	case "vulkan":
		return mb.DeviceVulkan, nil
	default:
		return 0, fmt.Errorf("unknown device %q", value)
	}
}

func loadDemoServer(libraryPath, modelPath, styleDirectory string, device mb.Device) (*demoServer, error) {
	library, err := mb.Open(libraryPath)
	if err != nil {
		return nil, fmt.Errorf("open native library: %w", err)
	}
	model, err := library.LoadModel(modelPath, device)
	if err != nil {
		library.Close()
		return nil, err
	}
	paths, err := filepath.Glob(filepath.Join(styleDirectory, "*.mbstyle"))
	if err != nil || len(paths) == 0 {
		model.Close()
		library.Close()
		return nil, fmt.Errorf("no .mbstyle files in %s", styleDirectory)
	}
	sort.Strings(paths)
	server := &demoServer{library: library, model: model, styles: make(map[string]*mb.Style), sessions: make(map[string]*session)}
	for _, path := range paths {
		style, loadErr := model.LoadStyle(path)
		if loadErr != nil {
			server.Close()
			return nil, fmt.Errorf("load %s: %w", path, loadErr)
		}
		if _, exists := server.styles[style.Name]; exists {
			style.Close()
			server.Close()
			return nil, fmt.Errorf("duplicate style %q", style.Name)
		}
		server.styles[style.Name] = style
		server.ordered = append(server.ordered, styleInfo{Name: style.Name, Speed: style.Speed})
	}
	sort.Slice(server.ordered, func(i, j int) bool { return server.ordered[i].Name < server.ordered[j].Name })
	server.joints, err = model.Skeleton()
	if err != nil {
		server.Close()
		return nil, err
	}
	root, err := fs.Sub(webFiles, "web")
	if err != nil {
		server.Close()
		return nil, err
	}
	server.static = http.FileServer(http.FS(root))
	return server, nil
}

func (s *demoServer) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	for _, item := range s.sessions {
		item.agent.Close()
	}
	s.sessions = nil
	s.mu.Unlock()
	for _, style := range s.styles {
		style.Close()
	}
	if s.model != nil {
		s.model.Close()
	}
	if s.library != nil {
		_ = s.library.Close()
	}
}

func randomID() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(value[:]), nil
}

func jsonResponse(writer http.ResponseWriter, status int, value any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.Header().Set("Cache-Control", "no-store")
	writer.WriteHeader(status)
	_ = json.NewEncoder(writer).Encode(value)
}
func apiError(writer http.ResponseWriter, status int, err error) {
	jsonResponse(writer, status, map[string]string{"error": err.Error()})
}
func decodeJSON(request *http.Request, output any) error {
	decoder := json.NewDecoder(io.LimitReader(request.Body, 1<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(output); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	return nil
}
func finite(values ...float32) bool {
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return false
		}
	}
	return true
}

func (s *demoServer) style(name string) (*mb.Style, error) {
	style := s.styles[name]
	if style == nil {
		return nil, fmt.Errorf("unknown style %q", name)
	}
	return style, nil
}

func (s *demoServer) commandPlan(item *session, style *mb.Style, request planRequest) (*mb.Motion, error) {
	if !finite(request.Move[0], request.Move[1], request.Facing[0], request.Facing[1]) {
		return nil, errors.New("control vector is not finite")
	}
	if math.Hypot(float64(request.Facing[0]), float64(request.Facing[1])) < 1e-6 {
		return nil, errors.New("facing vector is zero")
	}
	item.mu.Lock()
	defer item.mu.Unlock()
	if item.planned && request.Advance > 0 {
		if err := item.agent.Advance(request.Advance); err != nil {
			return nil, err
		}
	}
	command, err := s.library.NewCommand()
	if err != nil {
		return nil, err
	}
	defer command.Close()
	if err = command.SetStyle(style); err != nil {
		return nil, err
	}
	if err = command.SetMovement(request.Move[0], 0, request.Move[1]); err != nil {
		return nil, err
	}
	if err = command.SetFacing(request.Facing[0], 0, request.Facing[1]); err != nil {
		return nil, err
	}
	if request.Speed != nil {
		if !finite(*request.Speed) || *request.Speed < 0 {
			return nil, errors.New("speed is invalid")
		}
		if err = command.SetSpeed(*request.Speed); err != nil {
			return nil, err
		}
	}
	if err = command.SetSeed(request.Seed); err != nil {
		return nil, err
	}
	s.planMu.Lock()
	motion, err := item.agent.Plan(command)
	s.planMu.Unlock()
	if err == nil {
		item.planned = true
	}
	return motion, err
}

func (s *demoServer) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, _ *http.Request) {
		jsonResponse(w, http.StatusOK, map[string]any{"ok": true})
	})
	mux.HandleFunc("GET /api/meta", func(w http.ResponseWriter, _ *http.Request) {
		jsonResponse(w, http.StatusOK, map[string]any{"fps": 30, "joints": s.joints, "styles": s.ordered})
	})
	mux.HandleFunc("POST /api/session", s.createSession)
	mux.HandleFunc("POST /api/plan", s.plan)
	mux.Handle("/", s.static)
	return securityHeaders(mux)
}

func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'")
		next.ServeHTTP(w, r)
	})
}

func (s *demoServer) createSession(w http.ResponseWriter, r *http.Request) {
	var request sessionRequest
	if err := decodeJSON(r, &request); err != nil {
		apiError(w, http.StatusBadRequest, err)
		return
	}
	if request.Style == "" {
		if _, ok := s.styles["idle"]; ok {
			request.Style = "idle"
		} else {
			request.Style = s.ordered[0].Name
		}
	}
	style, err := s.style(request.Style)
	if err != nil {
		apiError(w, http.StatusBadRequest, err)
		return
	}
	agent, err := s.model.NewAgent()
	if err != nil {
		apiError(w, http.StatusInternalServerError, err)
		return
	}
	if err = agent.Reset(style); err != nil {
		agent.Close()
		apiError(w, http.StatusInternalServerError, err)
		return
	}
	id, err := randomID()
	if err != nil {
		agent.Close()
		apiError(w, http.StatusInternalServerError, err)
		return
	}
	item := &session{agent: agent}
	s.mu.Lock()
	s.sessions[id] = item
	s.mu.Unlock()
	motion, err := s.commandPlan(item, style, planRequest{Move: [2]float32{0, 0}, Facing: [2]float32{0, 1}, Seed: 1})
	if err != nil {
		s.mu.Lock()
		delete(s.sessions, id)
		s.mu.Unlock()
		agent.Close()
		apiError(w, http.StatusInternalServerError, err)
		return
	}
	jsonResponse(w, http.StatusOK, planResponse{Session: id, Style: style.Name, Motion: motion, Targets: motion.Targets})
}

func (s *demoServer) plan(w http.ResponseWriter, r *http.Request) {
	var request planRequest
	if err := decodeJSON(r, &request); err != nil {
		apiError(w, http.StatusBadRequest, err)
		return
	}
	s.mu.Lock()
	item := s.sessions[request.Session]
	s.mu.Unlock()
	if item == nil {
		apiError(w, http.StatusNotFound, errors.New("unknown session"))
		return
	}
	style, err := s.style(request.Style)
	if err != nil {
		apiError(w, http.StatusBadRequest, err)
		return
	}
	motion, err := s.commandPlan(item, style, request)
	if err != nil {
		apiError(w, http.StatusInternalServerError, err)
		return
	}
	jsonResponse(w, http.StatusOK, planResponse{Session: request.Session, Style: style.Name, Motion: motion, Targets: motion.Targets})
}

func main() {
	listen := flag.String("listen", "127.0.0.1:8080", "HTTP listen address")
	libraryPath := flag.String("library", os.Getenv("MOTIONBRICKS_LIB"), "path to libmotionbricks")
	modelPath := flag.String("model", os.Getenv("MOTIONBRICKS_MODEL"), "model bundle directory")
	stylesPath := flag.String("styles", os.Getenv("MOTIONBRICKS_STYLES"), "style directory")
	deviceName := flag.String("device", "cpu", "auto, cpu, or vulkan")
	flag.Parse()
	if *libraryPath == "" || *modelPath == "" || *stylesPath == "" {
		log.Fatal("-library, -model, and -styles are required")
	}
	device, err := parseDevice(*deviceName)
	if err != nil {
		log.Fatal(err)
	}
	demo, err := loadDemoServer(*libraryPath, *modelPath, *stylesPath, device)
	if err != nil {
		log.Fatal(err)
	}
	defer demo.Close()
	server := &http.Server{Addr: *listen, Handler: demo.routes(), ReadHeaderTimeout: 5 * time.Second}
	stopped := make(chan os.Signal, 1)
	signal.Notify(stopped, os.Interrupt, syscall.SIGTERM)
	go func() { <-stopped; _ = server.Close() }()
	log.Printf("MotionBricks demo: http://%s", *listen)
	if err = server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}
