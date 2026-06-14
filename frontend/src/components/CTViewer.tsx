import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  Brain,
  ChevronDown,
  ChevronUp,
  Download,
  Eye,
  EyeOff,
  Layers,
  Loader2,
  Maximize2,
  Minimize2,
  Play,
  RotateCcw,
} from "lucide-react";
import type { AnalysisStatus } from "../App";
import { Niivue, NVImage, SLICE_TYPE } from "@niivue/niivue";

interface Props {
  status?: AnalysisStatus;
  pid?: string | null;
  seriesUid?: string | null;
}

type Segmentation = {
  file: File;
  prompt: string;
  isVisible: boolean;
  color: string;
};

const BACKEND = import.meta.env.VITE_BACKEND_URL ?? "";

function cx(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function getSliceTypeLabel(sliceType: SLICE_TYPE) {
  if (sliceType === SLICE_TYPE.AXIAL) return "Axial";
  if (sliceType === SLICE_TYPE.CORONAL) return "Coronal";
  if (sliceType === SLICE_TYPE.SAGITTAL) return "Sagittal";
  if (sliceType === SLICE_TYPE.MULTIPLANAR) return "Multi";
  return "Viewer";
}

function SectionTitle({ children }: { children: ReactNode }) {
  return (
    <h2 className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">
      {children}
    </h2>
  );
}

function buildPatientUrl(path: string, pid: string, seriesUid: string) {
  return `${BACKEND}${path}/${encodeURIComponent(pid)}/${encodeURIComponent(seriesUid)}`;
}

async function readError(response: Response, fallback: string) {
  const text = await response.text().catch(() => "");
  if (!text) return fallback;

  try {
    const payload = JSON.parse(text);
    return payload.detail || payload.message || text;
  } catch {
    return text;
  }
}

export default function CTViewer({ status, pid, seriesUid }: Props) {
  const downloadMenuRef = useRef<HTMLDivElement>(null);

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [viewerKey, setViewerKey] = useState("empty");
  const [sliceType, setSliceType] = useState<SLICE_TYPE>(SLICE_TYPE.MULTIPLANAR);
  const [segmentation, setSegmentation] = useState<Segmentation | null>(null);

  const [prompt, setPrompt] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isLoadingVolume, setIsLoadingVolume] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showDownloadMenu, setShowDownloadMenu] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hasActivePatient = Boolean(pid && seriesUid);
  const activeViewLabel = getSliceTypeLabel(sliceType);

  useEffect(() => {
    if (!showDownloadMenu) return;

    const handleClick = (event: MouseEvent) => {
      if (
        downloadMenuRef.current &&
        !downloadMenuRef.current.contains(event.target as Node)
      ) {
        setShowDownloadMenu(false);
      }
    };

    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [showDownloadMenu]);

  useEffect(() => {
    if (!imageFile) return;

    const first = window.setTimeout(() => {
      window.dispatchEvent(new Event("resize"));
    }, 80);
    const second = window.setTimeout(() => {
      window.dispatchEvent(new Event("resize"));
    }, 260);

    return () => {
      window.clearTimeout(first);
      window.clearTimeout(second);
    };
  }, [imageFile, sliceType, isFullscreen]);

  useEffect(() => {
    if (!isFullscreen) return;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsFullscreen(false);
    };

    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [isFullscreen]);

  useEffect(() => {
    setPrompt("");
    setSegmentation(null);
    setShowDownloadMenu(false);
    setError(null);
    setSliceType(SLICE_TYPE.MULTIPLANAR);
    setIsFullscreen(false);

    if (!pid || !seriesUid) {
      setImageFile(null);
      setViewerKey("empty");
      return;
    }

    const controller = new AbortController();

    async function loadPatientVolume() {
      setIsLoadingVolume(true);
      setImageFile(null);
      setViewerKey("loading");

      try {
        const response = await fetch(
          buildPatientUrl("/api/v1/voxtell/volume", pid, seriesUid),
          { signal: controller.signal },
        );

        if (!response.ok) {
          throw new Error(
            await readError(response, "Không tải được CT volume từ backend."),
          );
        }

        const blob = await response.blob();
        const file = new File([blob], `${pid}_${seriesUid}.nii.gz`, {
          type: "application/gzip",
        });

        setImageFile(file);
        setViewerKey(`${pid}-${seriesUid}-${blob.size}-${Date.now()}`);
      } catch (err) {
        if (controller.signal.aborted) return;
        console.error("Error loading VoxTell volume:", err);
        setImageFile(null);
        setViewerKey("empty");
        setError(
          err instanceof Error
            ? err.message
            : `Không tải được CT volume. Kiểm tra backend VoxTell.`,
        );
      } finally {
        if (!controller.signal.aborted) setIsLoadingVolume(false);
      }
    }

    loadPatientVolume();

    return () => controller.abort();
  }, [pid, seriesUid]);

  const resetViewer = () => {
    setSegmentation(null);
    setPrompt("");
    setIsFullscreen(false);
    setShowDownloadMenu(false);
    setError(null);
  };

  const handleSegmentation = async () => {
    if (!pid || !seriesUid || !imageFile || !prompt.trim()) return;

    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("pid", pid);
      formData.append("series_uid", seriesUid);
      formData.append("prompt", prompt.trim());

      const response = await fetch(`${BACKEND}/api/v1/voxtell/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(
          await readError(response, "Không chạy được segmentation."),
        );
      }

      const blob = await response.blob();
      const file = new File(
        [blob],
        `voxtell_${pid}_${seriesUid}_${prompt.trim().replace(/\s+/g, "_")}.nii.gz`,
        { type: "application/gzip" },
      );

      setSegmentation({
        file,
        prompt: prompt.trim(),
        isVisible: true,
        color: "red",
      });
      setPrompt("");
    } catch (err) {
      console.error("Error running segmentation:", err);
      setError(
        err instanceof Error
          ? err.message
          : `Không chạy được segmentation. Kiểm tra backend VoxTell.`,
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadFile = (file: File, filename: string) => {
    const url = URL.createObjectURL(file);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  const handleDownload = () => {
    if (!segmentation) return;
    downloadFile(
      segmentation.file,
      `voxtell_${pid ?? "patient"}_${segmentation.prompt.replace(/\s+/g, "_")}.nii.gz`,
    );
    setShowDownloadMenu(false);
  };

  return (
    <section
      className={cx(
        "flex h-[calc(100vh-4.75rem)] min-h-[720px] w-[33.333vw] min-w-[380px] max-w-[640px] shrink-0 flex-col rounded-2xl border border-slate-800 bg-slate-950 font-sans text-slate-200 selection:bg-indigo-500/30",
        isFullscreen ? "overflow-visible" : "overflow-hidden",
      )}
    >
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
        <div className="border-b border-slate-800 bg-slate-900/80 p-4 backdrop-blur-xl">
          <div className="flex items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-3">
              <div className="rounded-lg bg-indigo-600 p-2 shadow-lg shadow-indigo-950/40">
                <Brain className="h-5 w-5 text-white" />
              </div>
              <div className="min-w-0">
                <h1 className="truncate text-lg font-bold text-white">
                  VoxTell CT Viewer
                </h1>
                <p className="text-[10px] font-medium uppercase tracking-[0.18em] text-slate-500">
                  Auto DICOM from S3 · NiiVue volume viewer
                </p>
              </div>
            </div>

            {status && (
              <span className="shrink-0 rounded-full border border-slate-700 bg-slate-950/70 px-2.5 py-1 text-[10px] font-mono text-slate-400">
                {status}
              </span>
            )}
          </div>

          <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/50 p-3">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-slate-900/70">
                {isLoadingVolume ? (
                  <Loader2 className="h-5 w-5 animate-spin text-indigo-400" />
                ) : (
                  <Layers
                    className={cx(
                      "h-5 w-5",
                      imageFile ? "text-indigo-400" : "text-slate-500",
                    )}
                  />
                )}
              </div>

              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-slate-200">
                  {hasActivePatient
                    ? isLoadingVolume
                      ? "Loading CT volume from S3..."
                      : imageFile
                        ? "CT volume ready"
                        : "CT volume not loaded"
                    : "No active patient"}
                </p>
                <p className="mt-1 break-all text-xs leading-relaxed text-slate-500">
                  {hasActivePatient
                    ? `pid=${pid} · series_uid=${seriesUid}`
                    : "Add or select a patient to auto-load DICOM from S3."}
                </p>
              </div>
            </div>
          </div>

          {error && (
            <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs leading-relaxed text-red-300">
              {error}
            </div>
          )}
        </div>

        <div className="space-y-3 border-b border-slate-800 bg-slate-950/80 p-4">
          <div className="space-y-2">
            <SectionTitle>Text Prompt</SectionTitle>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="VD: left ventricle, tumor, lung nodule..."
              className="h-16 w-full resize-none rounded-lg border border-slate-700 bg-slate-800/50 p-3 text-sm text-slate-200 outline-none transition placeholder:text-slate-600 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/30"
            />
            <button
              type="button"
              onClick={handleSegmentation}
              disabled={
                !pid ||
                !seriesUid ||
                !imageFile ||
                !prompt.trim() ||
                isProcessing ||
                isLoadingVolume
              }
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-3 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 fill-current" />
                  Run Segmentation
                </>
              )}
            </button>
          </div>
        </div>

        <div
          className={cx(
            "min-h-0 bg-gradient-to-b from-slate-950 to-slate-900 p-3",
            isFullscreen &&
            "fixed inset-0 z-[9999] h-screen w-screen max-w-none overflow-hidden bg-slate-950/95 p-4 backdrop-blur-xl",
          )}
        >
          <div
            className={cx(
              "flex h-[500px] min-h-[500px] flex-col overflow-hidden rounded-2xl border border-slate-800 bg-black shadow-2xl",
              isFullscreen &&
              "!h-full !min-h-0 !w-full rounded-xl border-slate-700",
            )}
          >
            <div className="flex shrink-0 items-center justify-between gap-3 border-b border-slate-800 bg-slate-900/90 px-3 py-2">
              <div className="min-w-0">
                <h2 className="truncate text-sm font-semibold text-slate-100">
                  {isFullscreen ? "See more clearly" : "VoxTell CT Viewer"}
                </h2>
                <p className="truncate text-[11px] text-slate-500">
                  {imageFile
                    ? `Current view: ${activeViewLabel}${isFullscreen ? " · Press Esc to close" : ""}`
                    : hasActivePatient
                      ? "Waiting for backend volume conversion"
                      : "No patient selected"}
                </p>
              </div>

              <div className="flex shrink-0 items-center gap-2">
                <button
                  type="button"
                  onClick={() => setIsFullscreen((prev) => !prev)}
                  disabled={!imageFile}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-indigo-400/40 bg-indigo-500/10 px-2.5 py-1.5 text-[11px] font-semibold text-indigo-200 transition hover:bg-indigo-500/20 disabled:cursor-not-allowed disabled:opacity-40"
                  title={
                    isFullscreen
                      ? "Close fullscreen viewer"
                      : "Open viewer in fullscreen"
                  }
                >
                  {isFullscreen ? (
                    <Minimize2 className="h-3.5 w-3.5" />
                  ) : (
                    <Maximize2 className="h-3.5 w-3.5" />
                  )}
                  {isFullscreen ? "Close" : "See more clearly"}
                </button>

                <button
                  type="button"
                  onClick={resetViewer}
                  disabled={!imageFile && !segmentation && !prompt}
                  className="inline-flex items-center gap-1 rounded-lg border border-slate-700 px-2 py-1.5 text-[11px] font-semibold text-slate-400 transition hover:border-slate-500 hover:text-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <RotateCcw className="h-3 w-3" />
                  Reset
                </button>
              </div>
            </div>

            <div className="relative min-h-0 flex-1">
              {imageFile ? (
                <Viewer
                  key={viewerKey}
                  image={imageFile}
                  segmentation={segmentation}
                  sliceType={sliceType}
                  onSliceTypeChange={setSliceType}
                />
              ) : (
                <div className="flex h-full min-h-[360px] flex-col items-center justify-center gap-4 bg-slate-950/60 p-6 text-center text-slate-500">
                  {isLoadingVolume ? (
                    <Loader2 className="h-14 w-14 animate-spin text-indigo-400/60" />
                  ) : (
                    <Layers className="h-14 w-14 opacity-25" />
                  )}
                  <div>
                    <p className="text-sm font-semibold text-slate-400">
                      {hasActivePatient ? "No CT volume loaded" : "No active patient"}
                    </p>
                    <p className="mt-1 text-xs leading-relaxed text-slate-600">
                      {hasActivePatient
                        ? "Backend sẽ tự tải dicom-raw/{pid}/{series_uid}.zip từ S3 và convert sang NIfTI tạm."
                        : "Add hoặc chọn patient để tự động load CT volume."}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="space-y-3 border-t border-slate-800 bg-slate-950/80 p-4">
          <div className="flex items-center justify-between gap-2">
            <SectionTitle>Current Segmentation</SectionTitle>
            <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-400">
              {segmentation ? "1" : "0"}
            </span>
          </div>

          {!segmentation ? (
            <div className="rounded-lg border border-slate-800 bg-slate-950/30 px-3 py-2 text-xs text-slate-500">
              Chưa có mask segmentation. Mỗi prompt mới sẽ thay thế mask hiện tại.
            </div>
          ) : (
            <div className="rounded-lg border border-slate-800 bg-slate-800/40 p-2.5 transition hover:border-slate-700">
              <div className="flex items-center justify-between gap-2">
                <div className="flex min-w-0 items-center gap-2.5">
                  <span className="h-2.5 w-2.5 shrink-0 rounded-full bg-red-500" />
                  <p
                    className="truncate text-sm font-medium text-slate-300"
                    title={segmentation.prompt}
                  >
                    {segmentation.prompt}
                  </p>
                </div>

                <button
                  type="button"
                  onClick={() =>
                    setSegmentation((prev) =>
                      prev ? { ...prev, isVisible: !prev.isVisible } : prev,
                    )
                  }
                  title={segmentation.isVisible ? "Hide" : "Show"}
                  className="rounded-md p-1.5 text-slate-500 transition hover:bg-slate-700 hover:text-slate-200"
                >
                  {segmentation.isVisible ? (
                    <Eye className="h-4 w-4" />
                  ) : (
                    <EyeOff className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>
          )}

          {segmentation && (
            <div className="relative" ref={downloadMenuRef}>
              <button
                type="button"
                onClick={handleDownload}
                className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-700 px-3 py-2.5 text-sm font-semibold text-slate-300 transition hover:border-slate-600 hover:bg-slate-800"
              >
                <Download className="h-4 w-4" />
                Download Segmentation (.nii.gz)
              </button>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

interface ViewerProps {
  image?: File | string | null;
  segmentation?: {
    file: File | string;
    color: string;
    isVisible: boolean;
  } | null;
  sliceType?: SLICE_TYPE;
  onSliceTypeChange?: (st: SLICE_TYPE) => void;
}

const SLICE_OPTIONS: { label: string; value: SLICE_TYPE }[] = [
  { label: "Axial", value: SLICE_TYPE.AXIAL },
  { label: "Coronal", value: SLICE_TYPE.CORONAL },
  { label: "Sagittal", value: SLICE_TYPE.SAGITTAL },
  { label: "Multi", value: SLICE_TYPE.MULTIPLANAR },
];

const SLICE_AXIS: Record<number, number> = {
  [SLICE_TYPE.AXIAL]: 2,
  [SLICE_TYPE.CORONAL]: 1,
  [SLICE_TYPE.SAGITTAL]: 0,
};

function Viewer({
  image,
  segmentation,
  sliceType = SLICE_TYPE.AXIAL,
  onSliceTypeChange,
}: ViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nv, setNv] = useState<Niivue | null>(null);
  const loadedSegRef = useRef(false);
  const [isToolbarVisible, setIsToolbarVisible] = useState(true);
  const [sliceFrac, setSliceFrac] = useState(0.5);
  const [totalSlices, setTotalSlices] = useState(0);
  const updateSliceInfoRef = useRef<() => void>(() => { });
  const sliceTypeRef = useRef(sliceType);
  sliceTypeRef.current = sliceType;
  const [winMin, setWinMin] = useState(0);
  const [winMax, setWinMax] = useState(1);
  const [winRange, setWinRange] = useState<[number, number]>([0, 1]);

  const handleResize = useCallback(() => {
    if (!containerRef.current || !canvasRef.current || !nv) return;

    const { clientWidth, clientHeight } = containerRef.current;
    const dpr = window.devicePixelRatio || 1;

    canvasRef.current.width = clientWidth * dpr;
    canvasRef.current.height = clientHeight * dpr;
    canvasRef.current.style.width = `${clientWidth}px`;
    canvasRef.current.style.height = `${clientHeight}px`;

    nv.resizeListener();
  }, [nv]);

  const updateSliceInfo = useCallback(() => {
    if (!nv || nv.volumes.length === 0) return;
    const vol = nv.volumes[0];
    const dims = vol.dims;
    if (!dims) return;
    const axis = SLICE_AXIS[sliceType as number];
    if (axis !== undefined) {
      setTotalSlices(dims[axis + 1] || 1);
    }
    const pos = nv.scene.crosshairPos;
    if (pos && axis !== undefined) {
      setSliceFrac(pos[axis]);
    }
  }, [nv, sliceType]);

  updateSliceInfoRef.current = updateSliceInfo;

  useEffect(() => {
    if (!canvasRef.current) return;

    const niivue = new Niivue({
      backColor: [0, 0, 0, 1],
      show3Dcrosshair: true,
    });

    niivue.attachToCanvas(canvasRef.current);

    if (sliceType === SLICE_TYPE.MULTIPLANAR) {
      niivue.setSliceType(SLICE_TYPE.MULTIPLANAR);
      niivue.setMultiplanarLayout(2);
      niivue.setMultiplanarPadPixels(0);
    } else {
      niivue.setSliceType(sliceType);
    }

    niivue.onImageLoaded = () => {
      if (niivue.volumes.length > 0) {
        const vol = niivue.volumes[0];
        setWinMin(vol.cal_min ?? vol.global_min ?? 0);
        setWinMax(vol.cal_max ?? vol.global_max ?? 1);
        setWinRange([vol.global_min ?? 0, vol.global_max ?? 1]);
      }
      updateSliceInfoRef.current();
    };

    niivue.onLocationChange = () => {
      const axis = SLICE_AXIS[sliceTypeRef.current as number];
      if (axis === undefined) return;
      const frac = niivue.scene.crosshairPos[axis];
      setSliceFrac(frac);
    };

    setNv(niivue);
  }, []);

  useEffect(() => {
    if (!containerRef.current || !nv) return;

    handleResize();

    const resizeObserver = new ResizeObserver(() => {
      handleResize();
    });

    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
    };
  }, [nv, handleResize]);

  useEffect(() => {
    if (!nv) return;

    if (sliceType === SLICE_TYPE.MULTIPLANAR) {
      nv.setSliceType(SLICE_TYPE.MULTIPLANAR);
      nv.setMultiplanarLayout(2);
      nv.setMultiplanarPadPixels(0);
    } else {
      nv.setSliceType(sliceType);
    }

    nv.updateGLVolume();
    handleResize();
    updateSliceInfo();
  }, [nv, sliceType, handleResize, updateSliceInfo]);

  useEffect(() => {
    if (!nv || !image) return;

    const loadVolume = async () => {
      try {
        nv.volumes = [];
        loadedSegRef.current = false;

        if (typeof image === "string") {
          await nv.loadVolumes([{ url: image }]);
        } else if (image instanceof File) {
          const nvImage = await NVImage.loadFromFile({
            file: image,
            name: image.name,
          });
          nv.addVolume(nvImage);
        }
      } catch (e) {
        console.error("Failed to load image:", e);
      }

      nv.updateGLVolume();
      handleResize();
      updateSliceInfoRef.current();

      if (nv.volumes.length > 0) {
        const vol = nv.volumes[0];
        setWinMin(vol.cal_min ?? vol.global_min ?? 0);
        setWinMax(vol.cal_max ?? vol.global_max ?? 1);
        setWinRange([vol.global_min ?? 0, vol.global_max ?? 1]);
      }
    };

    loadVolume();
  }, [nv, image, handleResize]);

  useEffect(() => {
    if (!nv || !image) return;

    const syncSegmentation = async () => {
      while (nv.volumes.length > 1) {
        nv.removeVolume(nv.volumes[nv.volumes.length - 1]);
      }
      loadedSegRef.current = false;

      if (!segmentation) {
        nv.updateGLVolume();
        return;
      }

      try {
        const opacity = segmentation.isVisible ? 0.5 : 0;
        let vol;

        if (typeof segmentation.file === "string") {
          vol = await NVImage.loadFromUrl({
            url: segmentation.file,
            colormap: segmentation.color,
            opacity,
            name: "voxtell_segmentation",
          });
        } else if (segmentation.file instanceof File) {
          vol = await NVImage.loadFromFile({
            file: segmentation.file,
            name: "voxtell_segmentation",
            colormap: segmentation.color,
            opacity,
          });
        }

        if (vol) {
          vol.name = "voxtell_segmentation";
          nv.addVolume(vol);
          loadedSegRef.current = true;
        }
      } catch (e) {
        console.error("Failed to load segmentation:", e);
      }

      nv.updateGLVolume();
    };

    syncSegmentation();
  }, [nv, segmentation, image]);

  const handleSliceChange = useCallback(
    (fraction: number) => {
      if (!nv || nv.volumes.length === 0) return;
      const axis = SLICE_AXIS[sliceType as number];
      if (axis === undefined) return;

      const pos = [...nv.scene.crosshairPos] as [number, number, number];
      pos[axis] = fraction;
      nv.scene.crosshairPos = pos;
      nv.updateGLVolume();
      setSliceFrac(fraction);
    },
    [nv, sliceType],
  );

  const handleWindowChange = useCallback(
    (newMin: number, newMax: number) => {
      if (!nv || nv.volumes.length === 0) return;
      const vol = nv.volumes[0];
      const step = (winRange[1] - winRange[0]) / 500;
      if (newMin >= newMax) return;
      if (newMax - newMin < step) return;
      vol.cal_min = newMin;
      vol.cal_max = newMax;
      nv.updateGLVolume();
      setWinMin(newMin);
      setWinMax(newMax);
    },
    [nv, winRange],
  );

  const isSingleAxis =
    sliceType !== SLICE_TYPE.MULTIPLANAR && sliceType !== SLICE_TYPE.RENDER;

  const currentSliceNum = Math.round(sliceFrac * Math.max(totalSlices - 1, 1)) + 1;
  const hasVolume = nv !== null && nv.volumes.length > 0;
  const winStep = (winRange[1] - winRange[0]) / 500 || 1;
  const rangeSpan = winRange[1] - winRange[0] || 1;
  const minPct = ((winMin - winRange[0]) / rangeSpan) * 100;
  const maxPct = ((winMax - winRange[0]) / rangeSpan) * 100;

  return (
    <div
      ref={containerRef}
      className="w-full h-full bg-black rounded-lg overflow-hidden border border-slate-700 shadow-xl"
      style={{ position: "relative" }}
    >
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
        }}
      />

      {image && (
        <button
          onClick={() => setIsToolbarVisible((prev) => !prev)}
          className="absolute top-3 right-3 z-20 p-1.5 bg-slate-800/80 backdrop-blur-sm border border-slate-700/60 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700/80 transition-all duration-200 shadow-md"
          title={isToolbarVisible ? "Hide controls" : "Show controls"}
        >
          {isToolbarVisible ? (
            <ChevronDown className="w-3.5 h-3.5" />
          ) : (
            <ChevronUp className="w-3.5 h-3.5" />
          )}
        </button>
      )}

      {image && (
        <div
          className={`
            absolute bottom-3 left-3 right-3 z-10 flex flex-col gap-2
            rounded-xl border border-slate-700/60 bg-slate-900/90 px-3 py-2 shadow-lg backdrop-blur-md
            transition-all duration-300 ease-in-out
            ${isToolbarVisible
              ? "opacity-100 translate-y-0"
              : "opacity-0 translate-y-4 pointer-events-none"
            }
          `}
        >
          {isSingleAxis && totalSlices > 1 && (
            <div className="flex min-w-0 items-center gap-2">
              <span className="w-16 shrink-0 text-center font-mono text-[10px] text-slate-400">
                {currentSliceNum}/{totalSlices}
              </span>
              <input
                type="range"
                min={0}
                max={1}
                step={1 / Math.max(totalSlices - 1, 1)}
                value={sliceFrac}
                onChange={(e) => handleSliceChange(parseFloat(e.target.value))}
                className="h-1 min-w-0 flex-1 cursor-pointer accent-indigo-500"
                title={`Slice ${currentSliceNum} of ${totalSlices}`}
              />
            </div>
          )}

          {hasVolume && (
            <div className="flex min-w-0 items-center gap-2">
              <span className="w-7 shrink-0 whitespace-nowrap font-mono text-[10px] text-slate-400">
                WL
              </span>
              <div className="relative flex h-6 min-w-0 flex-1 items-center">
                <div
                  className="absolute left-0 right-0 h-1.5 rounded-full"
                  style={{
                    background: `linear-gradient(to right, #000 ${minPct}%, #000 ${minPct}%, #fff ${maxPct}%, #fff ${maxPct}%)`,
                  }}
                />
                <input
                  type="range"
                  min={winRange[0]}
                  max={winRange[1]}
                  step={winStep}
                  value={winMin}
                  onChange={(e) =>
                    handleWindowChange(parseFloat(e.target.value), winMax)
                  }
                  className="windowing-thumb pointer-events-none absolute h-1.5 w-full cursor-pointer appearance-none bg-transparent"
                  style={{ zIndex: 2 }}
                  title={`Window min: ${Math.round(winMin)}`}
                />
                <input
                  type="range"
                  min={winRange[0]}
                  max={winRange[1]}
                  step={winStep}
                  value={winMax}
                  onChange={(e) =>
                    handleWindowChange(winMin, parseFloat(e.target.value))
                  }
                  className="windowing-thumb pointer-events-none absolute h-1.5 w-full cursor-pointer appearance-none bg-transparent"
                  style={{ zIndex: 3 }}
                  title={`Window max: ${Math.round(winMax)}`}
                />
              </div>
              <span className="w-20 shrink-0 truncate text-right font-mono text-[10px] text-slate-400">
                {Math.round(winMin)}..{Math.round(winMax)}
              </span>
            </div>
          )}

          <div className="grid grid-cols-4 gap-1 rounded-lg bg-slate-950/50 p-1">
            {SLICE_OPTIONS.map((opt) => {
              const isActive = sliceType === opt.value;
              return (
                <button
                  key={opt.label}
                  type="button"
                  onClick={() => onSliceTypeChange?.(opt.value)}
                  className={`
                    rounded-md px-2 py-1.5 text-[11px] font-semibold transition-all duration-200
                    ${isActive
                      ? "bg-indigo-600 text-white shadow-md shadow-indigo-900/40"
                      : "text-slate-400 hover:bg-slate-700/60 hover:text-white"
                    }
                  `}
                  title={opt.label}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
