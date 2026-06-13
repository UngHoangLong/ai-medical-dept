import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type ReactNode,
} from "react";
import {
  Brain,
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
  Trash2,
  Upload,
  ChevronDown,
} from "lucide-react";
import type { AnalysisStatus } from "../App";
import { Niivue, NVImage, SLICE_TYPE } from "@niivue/niivue";

interface Props {
  status?: AnalysisStatus;
}

type Segmentation = {
  id: string;
  file: File;
  prompt: string;
  isVisible: boolean;
  color: string;
  displayColor: string;
};

const VOXTELL_API_BASE =
  import.meta.env.VITE_VOXTELL_API_BASE_URL || "http://localhost:1711";

const SEGMENTATION_COLORS = [
  { nv: "red", css: "#ef4444" },
  { nv: "green", css: "#22c55e" },
  { nv: "blue", css: "#3b82f6" },
  { nv: "warm", css: "#f59e0b" },
  { nv: "cool", css: "#06b6d4" },
  { nv: "violet", css: "#8b5cf6" },
  { nv: "winter", css: "#0ea5e9" },
];

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

export default function CTViewer({ status }: Props) {
  const downloadMenuRef = useRef<HTMLDivElement>(null);

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [viewerKey, setViewerKey] = useState("empty");
  const [sliceType, setSliceType] = useState<SLICE_TYPE>(
    SLICE_TYPE.MULTIPLANAR,
  );
  const [segmentations, setSegmentations] = useState<Segmentation[]>([]);

  const [prompt, setPrompt] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);

  const [inputFormat, setInputFormat] = useState<"nifti" | "dicom">("nifti");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [showDownloadMenu, setShowDownloadMenu] = useState(false);

  const [isLoadingFile, setIsLoadingFile] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

    // NiiVue needs a resize event after the container changes size.
    // Fire twice because fullscreen CSS transition/layout can settle slightly later.
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

  const cleanupDicomSession = () => {
    if (!sessionId) return;
    fetch(`${VOXTELL_API_BASE}/session/${sessionId}`, {
      method: "DELETE",
    }).catch(() => undefined);
    setSessionId(null);
  };

  const resetStudyState = () => {
    setPrompt("");
    setSegmentations([]);
    setShowDownloadMenu(false);
    setError(null);
  };

  const handleFileUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";

    if (!file) return;

    const lowerName = file.name.toLowerCase();
    resetStudyState();
    cleanupDicomSession();
    setIsFullscreen(false);

    if (lowerName.endsWith(".zip")) {
      setInputFormat("dicom");
      setIsConverting(true);
      setImageFile(null);
      setViewerKey("empty");

      try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${VOXTELL_API_BASE}/convert`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const payload = await response
            .json()
            .catch(() => ({ detail: "Conversion failed" }));
          throw new Error(payload.detail || "Conversion failed");
        }

        const newSessionId = response.headers.get("X-Session-Id");
        setSessionId(newSessionId);

        const blob = await response.blob();
        const niftiFile = new File([blob], "converted.nii.gz", {
          type: "application/gzip",
        });

        setIsLoadingFile(true);
        setSliceType(SLICE_TYPE.MULTIPLANAR);
        setViewerKey(`${niftiFile.name}-${niftiFile.size}-${Date.now()}`);
        setImageFile(niftiFile);
        window.setTimeout(() => setIsLoadingFile(false), 500);
      } catch (err) {
        console.error("Error converting DICOM:", err);
        setInputFormat("nifti");
        setError(
          `Không convert được DICOM. Kiểm tra backend VoxTell tại ${VOXTELL_API_BASE}.`,
        );
      } finally {
        setIsConverting(false);
      }

      return;
    }

    const isNifti =
      lowerName.endsWith(".nii") ||
      lowerName.endsWith(".nii.gz") ||
      lowerName.endsWith(".gz");

    if (!isNifti) {
      setImageFile(null);
      setViewerKey("empty");
      setError(
        "File chưa đúng định dạng. Hãy upload .nii, .nii.gz hoặc .zip DICOM.",
      );
      return;
    }

    setInputFormat("nifti");
    setIsLoadingFile(true);
    setSliceType(SLICE_TYPE.MULTIPLANAR);
    setViewerKey(`${file.name}-${file.size}-${file.lastModified}`);
    setImageFile(file);

    window.setTimeout(() => {
      setIsLoadingFile(false);
    }, 500);
  };

  const resetViewer = () => {
    cleanupDicomSession();
    setImageFile(null);
    setViewerKey("empty");
    setSegmentations([]);
    setPrompt("");
    setIsFullscreen(false);
    setShowDownloadMenu(false);
    setError(null);
  };

  const handleSegmentation = async () => {
    if (!imageFile || !prompt.trim()) return;

    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("image", imageFile);
      formData.append("prompt", prompt.trim());

      const response = await fetch(`${VOXTELL_API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = await response
          .json()
          .catch(() => ({ detail: "Segmentation failed" }));
        throw new Error(payload.detail || "Segmentation failed");
      }

      const blob = await response.blob();
      const file = new File([blob], `segmentation_${imageFile.name}`, {
        type: "application/gzip",
      });
      const color =
        SEGMENTATION_COLORS[segmentations.length % SEGMENTATION_COLORS.length];

      setSegmentations((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          file,
          prompt: prompt.trim(),
          isVisible: true,
          color: color.nv,
          displayColor: color.css,
        },
      ]);
      setPrompt("");
    } catch (err) {
      console.error("Error running segmentation:", err);
      setError(
        `Không chạy được segmentation. Kiểm tra backend VoxTell tại ${VOXTELL_API_BASE}.`,
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const toggleVisibility = (id: string) => {
    setSegmentations((prev) =>
      prev.map((segmentation) =>
        segmentation.id === id
          ? { ...segmentation, isVisible: !segmentation.isVisible }
          : segmentation,
      ),
    );
  };

  const deleteSegmentation = (id: string) => {
    setSegmentations((prev) =>
      prev.filter((segmentation) => segmentation.id !== id),
    );
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
    if (segmentations.length === 0) return;

    if (segmentations.length === 1) {
      const segmentation = segmentations[0];
      downloadFile(
        segmentation.file,
        `voxtell_${segmentation.prompt.replace(/\s+/g, "_")}.nii.gz`,
      );
      return;
    }

    segmentations.forEach((segmentation, index) => {
      window.setTimeout(() => {
        downloadFile(
          segmentation.file,
          `voxtell_${index + 1}_${segmentation.prompt.replace(/\s+/g, "_")}.nii.gz`,
        );
      }, index * 100);
    });
  };

  const handleExportRtstruct = async () => {
    if (!sessionId || segmentations.length === 0) return;

    setIsExporting(true);
    setShowDownloadMenu(false);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append(
        "structure_names",
        JSON.stringify(
          segmentations.map((segmentation) => segmentation.prompt),
        ),
      );

      for (const segmentation of segmentations) {
        formData.append("segmentation_files", segmentation.file);
      }

      const response = await fetch(`${VOXTELL_API_BASE}/export-rtstruct`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = await response
          .json()
          .catch(() => ({ detail: "Export failed" }));
        throw new Error(payload.detail || "RTSTRUCT export failed");
      }

      const blob = await response.blob();
      const file = new File([blob], "rtstruct.dcm", {
        type: "application/dicom",
      });
      downloadFile(file, "rtstruct.dcm");
    } catch (err) {
      console.error("Error exporting RTSTRUCT:", err);
      setError("Không export được RTSTRUCT. Kiểm tra backend VoxTell.");
    } finally {
      setIsExporting(false);
    }
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
                  NIfTI volume viewer · segmentation mode
                </p>
              </div>
            </div>

            {status && (
              <span className="shrink-0 rounded-full border border-slate-700 bg-slate-950/70 px-2.5 py-1 text-[10px] font-mono text-slate-400">
                {status}
              </span>
            )}
          </div>

          <div className="mt-4 space-y-3">
            <div className="relative">
              <input
                type="file"
                accept=".nii,.nii.gz,.gz,.zip"
                onChange={handleFileUpload}
                className="absolute inset-0 z-20 h-full w-full cursor-pointer opacity-0"
              />

              <div
                className={cx(
                  "flex items-center gap-3 rounded-xl border border-dashed p-3 transition",
                  imageFile || isConverting
                    ? "border-indigo-500/50 bg-indigo-500/10"
                    : "border-slate-700 bg-slate-800/30 hover:border-slate-500 hover:bg-slate-800/50",
                )}
              >
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-slate-950/50">
                  {isConverting || isLoadingFile ? (
                    <Loader2 className="h-5 w-5 animate-spin text-indigo-400" />
                  ) : (
                    <Upload
                      className={cx(
                        "h-5 w-5",
                        imageFile ? "text-indigo-400" : "text-slate-400",
                      )}
                    />
                  )}
                </div>

                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-slate-200">
                    {isConverting
                      ? "Converting DICOM..."
                      : imageFile
                        ? imageFile.name
                        : "Upload CT volume"}
                  </p>
                  <p className="text-xs text-slate-500">
                    {isConverting
                      ? "Please wait"
                      : imageFile
                        ? `${(imageFile.size / 1024 / 1024).toFixed(1)} MB · ready`
                        : ".nii, .nii.gz hoặc .zip DICOM"}
                  </p>
                </div>
              </div>
            </div>

            {error && (
              <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs leading-relaxed text-red-300">
                {error}
              </div>
            )}
          </div>
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
                !imageFile || !prompt.trim() || isProcessing || isConverting
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
              isFullscreen && "!h-full !min-h-0 !w-full rounded-xl border-slate-700",
            )}
          >
            <div className="flex shrink-0 items-center justify-between gap-3 border-b border-slate-800 bg-slate-900/90 px-3 py-2">
              <div className="min-w-0">
                <h2 className="truncate text-sm font-semibold text-slate-100">
                  {isFullscreen ? "See more clearly" : "VoxTell CT Viewer"}
                </h2>
                <p className="truncate text-[11px] text-slate-500">
                  {imageFile
                    ? `${imageFile.name} · Current view: ${activeViewLabel}${isFullscreen ? " · Press Esc to close" : ""}`
                    : "Upload a NIfTI file to start visualization"}
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
                  disabled={!imageFile}
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
                  segmentations={segmentations}
                  sliceType={sliceType}
                  onSliceTypeChange={setSliceType}
                />
              ) : (
                <div className="flex h-full min-h-[360px] flex-col items-center justify-center gap-4 bg-slate-950/60 p-6 text-center text-slate-500">
                  <Layers className="h-14 w-14 opacity-25" />
                  <div>
                    <p className="text-sm font-semibold text-slate-400">
                      No CT volume loaded
                    </p>
                    <p className="mt-1 text-xs leading-relaxed text-slate-600">
                      Upload .nii hoặc .nii.gz để kiểm tra NiiVue viewer trước.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="space-y-3 border-t border-slate-800 bg-slate-950/80 p-4">
          <div className="flex items-center justify-between gap-2">
            <SectionTitle>Segmentations</SectionTitle>
            <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] text-slate-400">
              {segmentations.length}
            </span>
          </div>

          {segmentations.length === 0 ? (
            <div className="rounded-lg border border-slate-800 bg-slate-950/30 px-3 py-2 text-xs text-slate-500">
              Chưa có mask segmentation.
            </div>
          ) : (
            <div className="space-y-2">
              {segmentations.map((segmentation) => (
                <div
                  key={segmentation.id}
                  className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-800/40 p-2.5 transition hover:border-slate-700"
                >
                  <div className="flex min-w-0 items-center gap-2.5">
                    <span
                      className="h-2.5 w-2.5 shrink-0 rounded-full"
                      style={{ backgroundColor: segmentation.displayColor }}
                    />
                    <div className="min-w-0">
                      <p
                        className="truncate text-sm font-medium text-slate-300"
                        title={segmentation.prompt}
                      >
                        {segmentation.prompt}
                      </p>
                    </div>
                  </div>

                  <div className="flex shrink-0 items-center gap-0.5">
                    <button
                      type="button"
                      onClick={() => toggleVisibility(segmentation.id)}
                      title={segmentation.isVisible ? "Hide" : "Show"}
                      className="rounded-md p-1.5 text-slate-500 transition hover:bg-slate-700 hover:text-slate-200"
                    >
                      {segmentation.isVisible ? (
                        <Eye className="h-4 w-4" />
                      ) : (
                        <EyeOff className="h-4 w-4" />
                      )}
                    </button>
                    <button
                      type="button"
                      onClick={() => deleteSegmentation(segmentation.id)}
                      title="Delete"
                      className="rounded-md p-1.5 text-slate-500 transition hover:bg-red-500/10 hover:text-red-400"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {segmentations.length > 0 &&
            (inputFormat === "nifti" ? (
              <button
                type="button"
                onClick={handleDownload}
                className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-700 px-3 py-2.5 text-sm font-semibold text-slate-300 transition hover:border-slate-600 hover:bg-slate-800"
              >
                <Download className="h-4 w-4" />
                Download{" "}
                {segmentations.length === 1
                  ? "Segmentation"
                  : `All (${segmentations.length})`}
              </button>
            ) : (
              <div className="relative" ref={downloadMenuRef}>
                {showDownloadMenu && (
                  <div className="absolute bottom-full left-0 right-0 z-30 mb-2 overflow-hidden rounded-xl border border-slate-700 bg-slate-800 shadow-xl">
                    <button
                      type="button"
                      onClick={() => {
                        handleDownload();
                        setShowDownloadMenu(false);
                      }}
                      className="flex w-full items-center gap-2 px-4 py-3 text-sm text-slate-300 transition hover:bg-slate-700"
                    >
                      <Download className="h-4 w-4" />
                      NIfTI (.nii.gz)
                    </button>
                    <div className="border-t border-slate-700" />
                    <button
                      type="button"
                      onClick={handleExportRtstruct}
                      disabled={isExporting}
                      className="flex w-full items-center gap-2 px-4 py-3 text-sm text-slate-300 transition hover:bg-slate-700 disabled:opacity-50"
                    >
                      {isExporting ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Download className="h-4 w-4" />
                      )}
                      RTSTRUCT (.dcm)
                    </button>
                  </div>
                )}

                <button
                  type="button"
                  onClick={() => setShowDownloadMenu((prev) => !prev)}
                  disabled={isExporting}
                  className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-700 px-3 py-2.5 text-sm font-semibold text-slate-300 transition hover:border-slate-600 hover:bg-slate-800 disabled:opacity-50"
                >
                  {isExporting ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Exporting...
                    </>
                  ) : (
                    <>
                      <Download className="h-4 w-4" />
                      Download{" "}
                      {segmentations.length === 1
                        ? "Segmentation"
                        : `All (${segmentations.length})`}
                      <ChevronUp className="h-3 w-3" />
                    </>
                  )}
                </button>
              </div>
            ))}
        </div>
      </div>
    </section>
  );
}

interface ViewerProps {
  image?: File | string | null;
  segmentations?: Array<{
    id: string;
    file: File | string;
    color: string;
    isVisible: boolean;
  }>;
  sliceType?: SLICE_TYPE;
  onSliceTypeChange?: (st: SLICE_TYPE) => void;
}

const SLICE_OPTIONS: { label: string; value: SLICE_TYPE }[] = [
  { label: "Axial", value: SLICE_TYPE.AXIAL },
  { label: "Coronal", value: SLICE_TYPE.CORONAL },
  { label: "Sagittal", value: SLICE_TYPE.SAGITTAL },
  { label: "Multi", value: SLICE_TYPE.MULTIPLANAR },
];

// Which fraction axis each slice type controls
const SLICE_AXIS: Record<number, number> = {
  [SLICE_TYPE.AXIAL]: 2, // Z
  [SLICE_TYPE.CORONAL]: 1, // Y
  [SLICE_TYPE.SAGITTAL]: 0, // X
};

function Viewer({
  image,
  segmentations = [],
  sliceType = SLICE_TYPE.MULTIPLANAR,
  onSliceTypeChange,
}: ViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nv, setNv] = useState<Niivue | null>(null);
  const loadedSegIdsRef = useRef<Set<string>>(new Set());
  // Bottom toolbar visibility
  const [isToolbarVisible, setIsToolbarVisible] = useState(true);
  // Slice position fraction (0-1) for the slider
  const [sliceFrac, setSliceFrac] = useState(0.5);
  // Total number of slices for the current axis
  const [totalSlices, setTotalSlices] = useState(0);
  // Stable ref to updateSliceInfo so image-loading effect doesn't re-trigger on sliceType change
  const updateSliceInfoRef = useRef<() => void>(() => { });
  // Ref to current sliceType so onLocationChange avoids stale closures
  const sliceTypeRef = useRef(sliceType);
  sliceTypeRef.current = sliceType;
  // Windowing (gray intensity mapping) state
  const [winMin, setWinMin] = useState(0);
  const [winMax, setWinMax] = useState(1);
  const [winRange, setWinRange] = useState<[number, number]>([0, 1]); // [global_min, global_max]

  // Resize handler to eliminate gray bars
  const handleResize = useCallback(() => {
    if (!containerRef.current || !canvasRef.current || !nv) return;

    const { clientWidth, clientHeight } = containerRef.current;
    const dpr = window.devicePixelRatio || 1;

    // Set canvas size to match container exactly
    canvasRef.current.width = clientWidth * dpr;
    canvasRef.current.height = clientHeight * dpr;
    canvasRef.current.style.width = `${clientWidth}px`;
    canvasRef.current.style.height = `${clientHeight}px`;

    nv.resizeListener();
  }, [nv]);

  // Compute total slices for the current axis
  const updateSliceInfo = useCallback(() => {
    if (!nv || nv.volumes.length === 0) return;
    const vol = nv.volumes[0];
    const dims = vol.dims; // [3, nx, ny, nz, ...]
    if (!dims) return;
    const axis = SLICE_AXIS[sliceType as number];
    if (axis !== undefined) {
      setTotalSlices(dims[axis + 1] || 1); // dims is 1-indexed: dims[1]=nx, dims[2]=ny, dims[3]=nz
    }
    // Sync slider with current crosshair position
    const pos = nv.scene.crosshairPos;
    if (pos && axis !== undefined) {
      setSliceFrac(pos[axis]);
    }
  }, [nv, sliceType]);

  // Keep ref in sync with latest callback
  updateSliceInfoRef.current = updateSliceInfo;

  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return;

    const niivue = new Niivue({
      backColor: [0, 0, 0, 1],
      show3Dcrosshair: true,
    });

    niivue.attachToCanvas(canvasRef.current);
    niivue.setSliceType(niivue.sliceTypeMultiplanar);
    niivue.setMultiplanarLayout(2);
    niivue.setMultiplanarPadPixels(0);

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

    return () => {
      // Cleanup
    };
  }, []);

  // Set up ResizeObserver to handle container size changes
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

  // Effect to sync the slice type from props
  useEffect(() => {
    if (!nv) return;
    nv.setSliceType(sliceType);
    if (sliceType === SLICE_TYPE.MULTIPLANAR) {
      nv.setMultiplanarLayout(2); // GRID
      nv.setMultiplanarPadPixels(0);
    }
    nv.updateGLVolume();
    handleResize();
    updateSliceInfo();
  }, [nv, sliceType, handleResize, updateSliceInfo]);

  // Effect to handle base image
  useEffect(() => {
    if (!nv || !image) return;

    const loadVolume = async () => {
      try {
        nv.volumes = [];
        loadedSegIdsRef.current.clear();

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

      // Initialize windowing from loaded volume
      if (nv.volumes.length > 0) {
        const vol = nv.volumes[0];
        setWinMin(vol.cal_min ?? vol.global_min ?? 0);
        setWinMax(vol.cal_max ?? vol.global_max ?? 1);
        setWinRange([vol.global_min ?? 0, vol.global_max ?? 1]);
      }
    };

    loadVolume();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nv, image, handleResize]);

  // Slice navigation handler — called from slider
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

  // Windowing change handler — sets cal_min/cal_max on the base volume
  const handleWindowChange = useCallback(
    (newMin: number, newMax: number) => {
      if (!nv || nv.volumes.length === 0) return;
      const vol = nv.volumes[0];
      // Prevent crossing: enforce at least a tiny gap
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

  // Effect to handle segmentations
  // Uses name-based volume matching instead of fragile numeric indices
  useEffect(() => {
    if (!nv || !image) return;

    // Helper: find a NiiVue volume by the segmentation id (stored as vol.name)
    const findVolumeBySegId = (segId: string) => {
      // Skip index 0 (the base image)
      for (let i = 1; i < nv.volumes.length; i++) {
        if (nv.volumes[i].name === segId) {
          return nv.volumes[i];
        }
      }
      return null;
    };

    const syncSegmentations = async () => {
      const currentIds = new Set(segmentations.map((s) => s.id));
      const loadedIds = new Set(loadedSegIdsRef.current);

      // 1. Remove segmentations that are no longer in props
      const idsToRemove = [...loadedIds].filter((id) => !currentIds.has(id));
      for (const id of idsToRemove) {
        const vol = findVolumeBySegId(id);
        if (vol) {
          nv.removeVolume(vol);
        }
        loadedSegIdsRef.current.delete(id);
      }

      // 2. Add new segmentations and update visibility for existing ones
      for (const seg of segmentations) {
        if (loadedSegIdsRef.current.has(seg.id)) {
          // Already loaded — just sync visibility
          const vol = findVolumeBySegId(seg.id);
          if (vol) {
            const newOpacity = seg.isVisible ? 0.5 : 0;
            if (vol.opacity !== newOpacity) {
              vol.opacity = newOpacity;
            }
          }
        } else {
          // New segmentation — load it
          try {
            let vol;
            const opacity = seg.isVisible ? 0.5 : 0;
            const colormap = seg.color;

            if (typeof seg.file === "string") {
              vol = await NVImage.loadFromUrl({
                url: seg.file,
                colormap: colormap,
                opacity: opacity,
                name: seg.id, // tag with seg id for reliable lookup
              });
            } else if (seg.file instanceof File) {
              vol = await NVImage.loadFromFile({
                file: seg.file,
                name: seg.id,
                colormap: colormap,
                opacity: opacity,
              });
            }

            if (vol) {
              // Ensure the name matches the seg id for lookup
              vol.name = seg.id;
              nv.addVolume(vol);
              loadedSegIdsRef.current.add(seg.id);
            }
          } catch (e) {
            console.error("Failed to load seg:", seg.id, e);
          }
        }
      }

      nv.updateGLVolume();
    };

    syncSegmentations();
  }, [nv, segmentations, image]);

  // Determine current slice state
  const isSingleAxis =
    sliceType !== SLICE_TYPE.MULTIPLANAR && sliceType !== SLICE_TYPE.RENDER;
  const currentSliceNum =
    Math.round(sliceFrac * Math.max(totalSlices - 1, 1)) + 1;

  // Windowing slider computed values
  const hasVolume = nv !== null && nv.volumes.length > 0;
  const winStep = (winRange[1] - winRange[0]) / 500 || 1;
  // Gradient positions as percentages for the track background
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

      {/* Toolbar toggle button */}
      {image && (
        <button
          onClick={() => setIsToolbarVisible((prev) => !prev)}
          className="absolute top-3 right-3 z-20 p-1.5 bg-slate-800/80 backdrop-blur-sm
                               border border-slate-700/60 rounded-lg text-slate-400
                               hover:text-white hover:bg-slate-700/80 transition-all duration-200 shadow-md"
          title={isToolbarVisible ? "Hide controls" : "Show controls"}
        >
          {isToolbarVisible ? (
            <ChevronDown className="w-3.5 h-3.5" />
          ) : (
            <ChevronUp className="w-3.5 h-3.5" />
          )}
        </button>
      )}

      {/* Bottom control bar */}
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
          {/* Slice slider — visible only in single-axis views */}
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
                onChange={(e) =>
                  handleSliceChange(parseFloat(e.target.value))
                }
                className="h-1 min-w-0 flex-1 cursor-pointer accent-indigo-500"
                title={`Slice ${currentSliceNum} of ${totalSlices}`}
              />
            </div>
          )}

          {/* Windowing slider — stacked row to avoid overlap in narrow CT panel */}
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

          {/* Layout / Axis Selector — own row so labels never overwrite sliders */}
          <div className="grid grid-cols-4 gap-1 rounded-lg bg-slate-950/50 p-1">
            {SLICE_OPTIONS.map((opt) => {
              const isActive = sliceType === opt.value;
              return (
                <button
                  key={opt.label}
                  type="button"
                  onClick={() => {
                    onSliceTypeChange?.(opt.value);
                  }}
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
