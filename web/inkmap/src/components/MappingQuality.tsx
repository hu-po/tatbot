import { useMemo } from "react";
import { CHART_AREA_BUDGET, CHART_CLOSURE_LIMIT, buildDecal } from "../core/decal.ts";
import { useStore } from "../store.ts";

export function MappingQuality() {
  const show = useStore(state => state.showQuality);
  const body = useStore(state => state.body);
  const selected = useStore(state => state.selected);
  const placement = useStore(state => state.placements.find(item => item.id === state.selected));
  const atlas = useStore(state => state.atlas);
  const result = useMemo(() => {
    if (!show || !body || !placement) return null;
    try {
      const built = buildDecal(body.restSkin.geometry, body.skin.geometry, {
        anchor: placement.anchor,
        rotationRad: placement.rotation_rad,
        sizeMm: placement.size_mm,
      });
      built.geometry.dispose();
      return { quality: built.quality, error: null };
    } catch (error) {
      return { quality: null, error: error instanceof Error ? error.message : String(error) };
    }
  }, [body, placement, show]);
  if (!show) return null;
  if (!selected || !placement) return <aside className="quality-overlay" aria-label="Mapping quality"><strong>Mapping quality</strong><span>Select a tattoo to inspect its chart.</span></aside>;
  if (!result?.quality) return <aside className="quality-overlay warning" aria-label="Mapping quality"><strong>Mapping refused</strong><span>{result?.error}</span></aside>;
  const q = result.quality;
  const areaDrift = Math.abs(q.mappedPercent - 100);
  const stretch = 100 * (q.areaStretchRatio - 1);
  const support = Boolean(atlas?.isValidAnchor(placement.anchor));
  const nearingLimit = areaDrift > CHART_AREA_BUDGET * 100 * 0.75 || q.seamRatio > CHART_CLOSURE_LIMIT * 0.75;
  return (
    <aside className={`quality-overlay${!support || nearingLimit ? " warning" : ""}`} aria-label="Mapping quality">
      <strong>{support ? "Supported mapping" : "Unsupported region"}</strong>
      <dl>
        <dt>mapped area</dt><dd>{q.mappedAreaMm2.toFixed(0)} mm² · {q.mappedPercent.toFixed(1)}%</dd>
        <dt>surface stretch</dt><dd>{stretch >= 0 ? "+" : ""}{stretch.toFixed(1)}% area</dd>
        <dt>chart overlap</dt><dd>{(q.seamRatio * 100).toFixed(1)}% of diagonal</dd>
        <dt>surface faces</dt><dd>{q.chartFaceCount}</dd>
        <dt>site</dt><dd>{[placement.site?.laterality, placement.site?.id.replaceAll("_", " ")].filter(Boolean).join(" ") || "unlabeled"}</dd>
      </dl>
      {nearingLimit && <span>Near the intrinsic-chart refusal limit; inspect before export.</span>}
    </aside>
  );
}
