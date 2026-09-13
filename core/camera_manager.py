import cv2
import time
import threading
import numpy as np
from typing import Optional, Tuple


class CameraManager:
    """
    Thread-safe hardware camera manager.

    Important:
    - Uses AVFoundation on macOS.
    - Capture thread owns VideoCapture operations.
    - stop() waits for the capture thread to finish BEFORE releasing
      the VideoCapture object.
    - Prevents Streamlit reruns from creating overlapping camera threads.
    """

    _instance = None
    _instance_lock = threading.RLock()

    def __new__(cls, src=0):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super(CameraManager, cls).__new__(cls)

                cls._instance.src = src
                cls._instance.cap = None

                cls._instance.running = False
                cls._instance.ret = False
                cls._instance.frame = None

                cls._instance.fps = 0.0
                cls._instance.prev_time = time.time()
                cls._instance.frame_count = 0

                cls._instance.thread = None

                # Protects frame/metrics shared between threads.
                cls._instance.frame_lock = threading.Lock()

                # Protects camera lifecycle.
                cls._instance.camera_lock = threading.RLock()

                cls._instance.start()

            elif not cls._instance.running:
                cls._instance.start()

            return cls._instance

    # ------------------------------------------------------------------
    # START
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Safely start the hardware camera."""

        with self.camera_lock:

            # Already running.
            if (
                self.running
                and self.thread is not None
                and self.thread.is_alive()
                and self.cap is not None
            ):
                return

            # Clean up a stale thread reference.
            if self.thread is not None and self.thread.is_alive():
                print("[CameraManager] Waiting for previous capture thread...")
                self.running = False

                self.thread.join(timeout=3.0)

                if self.thread.is_alive():
                    print(
                        "[CameraManager] WARNING: Previous camera thread "
                        "did not terminate cleanly."
                    )
                    return

                self.thread = None

            # Make absolutely sure an old capture object is gone.
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass

                self.cap = None

            self.ret = False

            with self.frame_lock:
                self.frame = None

            self.fps = 0.0
            self.prev_time = time.time()
            self.frame_count = 0

            print("[CameraManager] Opening hardware camera...")

            # ----------------------------------------------------------
            # macOS uses AVFoundation.
            # ----------------------------------------------------------

            self.cap = cv2.VideoCapture(
                self.src,
                cv2.CAP_AVFOUNDATION
            )

            # Fallback if AVFoundation failed.
            if not self.cap.isOpened():
                print(
                    "[CameraManager] AVFoundation failed. "
                    "Trying default OpenCV camera backend..."
                )

                try:
                    self.cap.release()
                except Exception:
                    pass

                self.cap = cv2.VideoCapture(self.src)

            if not self.cap.isOpened():
                print("[CameraManager] ERROR: Could not open camera.")

                self.cap = None
                self.running = False
                return

            # Camera configuration.
            try:
                self.cap.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    640
                )

                self.cap.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    480
                )

                self.cap.set(
                    cv2.CAP_PROP_FPS,
                    30
                )
            except Exception as e:
                print(
                    f"[CameraManager] Camera configuration warning: {e}"
                )

            self.running = True

            self.thread = threading.Thread(
                target=self._capture_loop,
                name="TARA-CameraCapture",
                daemon=True
            )

            self.thread.start()

            print(
                "[CameraManager] Hardware camera capture thread started."
            )

    # ------------------------------------------------------------------
    # CAPTURE LOOP
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """
        Camera capture thread.

        IMPORTANT:
        This thread is the ONLY thread that calls cap.read().
        """

        # Take a local reference.

        # This prevents the loop from repeatedly accessing
        # self.cap while another thread is changing the manager state.
        cap = self.cap

        if cap is None:
            print("[CameraManager] Capture loop started without camera.")
            return

        try:

            while self.running:

                # ------------------------------------------------------
                # Read frame.
                #
                # Do NOT release the camera from another thread while
                # this operation is happening.
                # stop() waits for this thread to finish.
                # ------------------------------------------------------

                ret, frame = cap.read()

                if not self.running:
                    break

                if ret and frame is not None:

                    with self.frame_lock:
                        self.ret = True

                        # Store a copy so that downstream processing
                        # doesn't accidentally mutate the camera buffer.
                        self.frame = frame.copy()

                    # FPS calculation.
                    now = time.time()
                    dt = now - self.prev_time

                    if dt >= 1.0:

                        self.fps = round(
                            self.frame_count / dt,
                            1
                        )

                        self.frame_count = 0
                        self.prev_time = now

                    else:
                        self.frame_count += 1

                else:

                    with self.frame_lock:
                        self.ret = False

                    # Camera temporarily failed to produce a frame.
                    time.sleep(0.01)

        except Exception as e:

            print(
                f"[CameraManager] Capture thread error: {e}"
            )

        finally:

            # IMPORTANT:
            # Do NOT call cap.release() here.
            #
            # stop() owns the final release operation and will only
            # execute it AFTER this thread has terminated.
            pass

        print(
            "[CameraManager] Capture thread exited safely."
        )

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------

    def read(
        self
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the latest camera frame safely."""

        if not self.running:
            return False, None

        with self.frame_lock:

            if self.frame is None:
                return False, None

            # Return a copy so callers cannot mutate the shared frame.
            return self.ret, self.frame.copy()

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------

    def update_frame_metrics(self) -> None:
        """Kept for compatibility with existing TARA code."""
        pass

    def get_status(self) -> dict:

        with self.frame_lock:
            ret = self.ret

        return {
            "fps": self.fps,
            "running": self.running,
            "connected": ret or self.running,
            "frame_count": self.frame_count + 1
        }

    # ------------------------------------------------------------------
    # STOP
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """
        Safely stop the camera.

        CRITICAL ORDER:

            1. Tell capture thread to stop.
            2. Wait for capture thread to exit.
            3. ONLY THEN release VideoCapture.
            4. Clear frame buffer.

        This prevents cap.release() from racing with cap.read().
        """

        with self.camera_lock:

            # Already stopped.
            if (
                not self.running
                and self.thread is None
                and self.cap is None
            ):
                return

            print(
                "[CameraManager] Stopping hardware camera..."
            )

            # ----------------------------------------------------------
            # STEP 1
            # Tell capture thread to stop.
            # ----------------------------------------------------------

            self.running = False

            thread = self.thread

            # ----------------------------------------------------------
            # STEP 2
            # Wait for capture thread to finish.
            # ----------------------------------------------------------

            if (
                thread is not None
                and thread.is_alive()
                and thread is not threading.current_thread()
            ):

                thread.join(timeout=3.0)

                if thread.is_alive():

                    print(
                        "[CameraManager] WARNING: "
                        "Camera thread did not stop within 3 seconds."
                    )

                    # DO NOT release cap here.
                    #
                    # Releasing it while the thread is still inside
                    # cap.read() could recreate the segmentation fault.
                    return

            # Thread is now safely finished.
            self.thread = None

            # ----------------------------------------------------------
            # STEP 3
            # ONLY NOW release VideoCapture.
            # ----------------------------------------------------------

            cap = self.cap
            self.cap = None

            if cap is not None:

                try:
                    cap.release()
                except Exception as e:
                    print(
                        f"[CameraManager] Camera release warning: {e}"
                    )

            # ----------------------------------------------------------
            # STEP 4
            # Clear frame buffer.
            # ----------------------------------------------------------

            with self.frame_lock:
                self.ret = False
                self.frame = None

            self.frame_count = 0
            self.fps = 0.0

            print(
                "[CameraManager] Hardware camera stopped "
                "and frame buffer purged."
            )