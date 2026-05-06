//go:build windows || darwin

package updater

import (
	"archive/zip"
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/app/store"
)

func TestIsNewReleaseAvailable(t *testing.T) {
	slog.SetLogLoggerLevel(slog.LevelDebug)
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
			// TODO - wire up the redirects to mimic real behavior
		} else {
			slog.Debug("unexpected request", "url", r.URL)
		}
	}))
	defer server.Close()
	slog.Debug("server", "url", server.URL)

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close() // Ensure database is closed
	UpdateCheckURLBase = server.URL + "/update.json"
	updatePresent, resp := updater.checkForUpdate(t.Context())
	if !updatePresent {
		t.Fatal("expected update to be available")
	}
	if resp.UpdateVersion != "9.9.9" {
		t.Fatal("unexpected response", "url", resp.UpdateURL, "version", resp.UpdateVersion)
	}
}

func TestBackgoundChecker(t *testing.T) {
	UpdateStageDir = t.TempDir()
	haveUpdate := atomic.Bool{}
	verified := atomic.Bool{}
	updateDone := make(chan struct{}, 1)
	cb := func(ver string) error {
		haveUpdate.Store(true)
		select {
		case updateDone <- struct{}{}:
		default:
		}
		return nil
	}
	stallTimer := time.NewTimer(5 * time.Second)
	defer stallTimer.Stop()
	ctx, cancel := context.WithCancel(t.Context())
	UpdateCheckInitialDelay = 5 * time.Millisecond
	UpdateCheckInterval = 5 * time.Millisecond
	setVerifyDownload(t, func() error {
		verified.Store(true)
		return nil
	})

	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
			// TODO - wire up the redirects to mimic real behavior
		} else if r.URL.Path == "/9.9.9/"+Installer {
			buf := &bytes.Buffer{}
			zw := zip.NewWriter(buf)
			zw.Close()
			io.Copy(w, buf)
		} else {
			slog.Debug("unexpected request", "url", r.URL)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()

	settings, err := updater.Store.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = true
	if err := updater.Store.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	stopped := updater.StartBackgroundUpdaterChecker(ctx, cb)
	defer waitForBackgroundUpdater(t, cancel, stopped)
	select {
	case <-stallTimer.C:
		t.Fatal("stalled")
	case <-updateDone:
		if !haveUpdate.Load() {
			t.Fatal("no update received")
		}
		if !verified.Load() {
			t.Fatal("unverified")
		}
	}
}

func TestAutoUpdateDisabledSkipsDownload(t *testing.T) {
	UpdateStageDir = t.TempDir()
	var downloadAttempted atomic.Bool
	var callbackCalled atomic.Bool

	ctx, cancel := context.WithCancel(t.Context())
	UpdateCheckInitialDelay = 5 * time.Millisecond
	UpdateCheckInterval = 5 * time.Millisecond
	setVerifyDownload(t, func() error {
		return nil
	})

	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
		} else if r.URL.Path == "/9.9.9/"+Installer {
			downloadAttempted.Store(true)
			buf := &bytes.Buffer{}
			zw := zip.NewWriter(buf)
			zw.Close()
			io.Copy(w, buf)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()

	// Ensure auto-update is disabled
	settings, err := updater.Store.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = false
	if err := updater.Store.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	cb := func(ver string) error {
		callbackCalled.Store(true)
		return nil
	}

	stopped := updater.StartBackgroundUpdaterChecker(ctx, cb)

	// Wait enough time for multiple check cycles
	time.Sleep(50 * time.Millisecond)
	waitForBackgroundUpdater(t, cancel, stopped)

	if downloadAttempted.Load() {
		t.Fatal("download should not be attempted when auto-update is disabled")
	}
	if callbackCalled.Load() {
		t.Fatal("callback should not be called when auto-update is disabled")
	}
}

func TestAutoUpdateReenabledDownloadsUpdate(t *testing.T) {
	UpdateStageDir = t.TempDir()
	var downloadAttempted atomic.Bool
	callbackCalled := make(chan struct{}, 1)

	ctx, cancel := context.WithCancel(t.Context())
	UpdateCheckInitialDelay = 5 * time.Millisecond
	UpdateCheckInterval = 5 * time.Millisecond
	setVerifyDownload(t, func() error {
		return nil
	})

	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
		} else if r.URL.Path == "/9.9.9/"+Installer {
			downloadAttempted.Store(true)
			buf := &bytes.Buffer{}
			zw := zip.NewWriter(buf)
			zw.Close()
			io.Copy(w, buf)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	upd := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer upd.Store.Close()

	// Start with auto-update disabled
	settings, err := upd.Store.Settings()
	if err != nil {
		t.Fatal(err)
	}
	settings.AutoUpdateEnabled = false
	if err := upd.Store.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	cb := func(ver string) error {
		select {
		case callbackCalled <- struct{}{}:
		default:
		}
		return nil
	}

	stopped := upd.StartBackgroundUpdaterChecker(ctx, cb)
	defer waitForBackgroundUpdater(t, cancel, stopped)

	// Wait for a few cycles with auto-update disabled - no download should happen
	time.Sleep(50 * time.Millisecond)
	if downloadAttempted.Load() {
		t.Fatal("download should not happen while auto-update is disabled")
	}

	// Re-enable auto-update
	settings.AutoUpdateEnabled = true
	if err := upd.Store.SetSettings(settings); err != nil {
		t.Fatal(err)
	}

	// Wait for the checker to pick it up and download
	select {
	case <-callbackCalled:
		// Success: download happened and callback was called after re-enabling
		if !downloadAttempted.Load() {
			t.Fatal("expected download to be attempted after re-enabling")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("expected download and callback after re-enabling auto-update")
	}
}

func TestCancelOngoingDownload(t *testing.T) {
	UpdateStageDir = t.TempDir()
	downloadStarted := make(chan struct{})
	downloadCancelled := make(chan struct{})

	ctx := t.Context()
	setVerifyDownload(t, func() error {
		return nil
	})

	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			w.Write([]byte(
				fmt.Sprintf(`{"version": "9.9.9", "url": "%s"}`,
					server.URL+"/9.9.9/"+Installer)))
		} else if r.URL.Path == "/9.9.9/"+Installer {
			if r.Method == http.MethodHead {
				w.Header().Set("Content-Length", "1000000")
				w.WriteHeader(http.StatusOK)
				return
			}
			// Signal that download has started
			close(downloadStarted)
			// Wait for cancellation or timeout
			select {
			case <-r.Context().Done():
				close(downloadCancelled)
				return
			case <-time.After(5 * time.Second):
				t.Error("download was not cancelled in time")
			}
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()

	_, resp := updater.checkForUpdate(ctx)

	// Start download in goroutine
	go func() {
		_ = updater.DownloadNewRelease(ctx, resp)
	}()

	// Wait for download to start
	select {
	case <-downloadStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("download did not start in time")
	}

	// Cancel the download
	updater.CancelOngoingDownload()

	// Verify cancellation was received
	select {
	case <-downloadCancelled:
		// Success
	case <-time.After(2 * time.Second):
		t.Fatal("download cancellation was not received by server")
	}
}

func TestTriggerImmediateCheck(t *testing.T) {
	UpdateStageDir = t.TempDir()
	checkCount := atomic.Int32{}
	checkDone := make(chan struct{}, 10)

	ctx, cancel := context.WithCancel(t.Context())
	// Set a very long interval so only TriggerImmediateCheck causes checks
	UpdateCheckInitialDelay = 1 * time.Millisecond
	UpdateCheckInterval = 1 * time.Hour
	setVerifyDownload(t, func() error {
		return nil
	})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/update.json" {
			checkCount.Add(1)
			select {
			case checkDone <- struct{}{}:
			default:
			}
			// Return no update available
			w.WriteHeader(http.StatusNoContent)
		}
	}))
	defer server.Close()
	UpdateCheckURLBase = server.URL + "/update.json"

	updater := &Updater{Store: &store.Store{DBPath: filepath.Join(t.TempDir(), "test.db")}}
	defer updater.Store.Close()

	cb := func(ver string) error {
		return nil
	}

	stopped := updater.StartBackgroundUpdaterChecker(ctx, cb)
	defer waitForBackgroundUpdater(t, cancel, stopped)

	// Wait for the initial check that fires after the initial delay
	select {
	case <-checkDone:
	case <-time.After(2 * time.Second):
		t.Fatal("initial check did not happen")
	}

	initialCount := checkCount.Load()

	// Trigger immediate check
	updater.TriggerImmediateCheck()

	// Wait for the triggered check
	select {
	case <-checkDone:
	case <-time.After(2 * time.Second):
		t.Fatal("triggered check did not happen")
	}

	finalCount := checkCount.Load()
	if finalCount <= initialCount {
		t.Fatalf("TriggerImmediateCheck did not cause additional check: initial=%d, final=%d", initialCount, finalCount)
	}
}

func waitForBackgroundUpdater(t *testing.T, cancel context.CancelFunc, stopped <-chan struct{}) {
	t.Helper()
	cancel()
	select {
	case <-stopped:
	case <-time.After(2 * time.Second):
		t.Fatal("background updater did not stop")
	}
}

func setVerifyDownload(t *testing.T, fn func() error) {
	t.Helper()
	old := VerifyDownload
	VerifyDownload = fn
	t.Cleanup(func() {
		VerifyDownload = old
	})
}
