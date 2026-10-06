import { SKIN_TONES } from "../core/defaults.ts";
import { editorPoseOptions } from "../core/pose.ts";
import { useStore } from "../store.ts";

/** Pose and skin tone, the two things about the body a person sets before
 *  placing anything. One markup, shown in the body panel and in the View
 *  menu alike. */
export function PosePicker() {
  const poseId = useStore((state) => state.poseId);
  const setPoseId = useStore((state) => state.setPoseId);
  return (
    <label className="menu-field">Pose<select aria-label="Pose" value={poseId} onChange={(event) => setPoseId(event.target.value)}>
      {editorPoseOptions(poseId).map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
    </select></label>
  );
}

export function SkinTonePicker() {
  const skinTone = useStore((state) => state.skinTone);
  const setSkinTone = useStore((state) => state.setSkinTone);
  const isActive = (hex: string) => hex.toLowerCase() === skinTone.toLowerCase();
  return (
    <div className="tones" role="radiogroup" aria-label="skin tone">
      {SKIN_TONES.map((hex) => <button key={hex} type="button" role="radio" aria-checked={isActive(hex)} className={isActive(hex) ? "tone active" : "tone"} style={{ background: hex }} title={hex} onClick={() => setSkinTone(hex)} />)}
      <label className="tone custom" title="custom colour" style={{ background: skinTone }}>
        <input type="color" value={skinTone} onChange={(event) => setSkinTone(event.target.value)} aria-label="custom skin tone" />
        <span>+</span>
      </label>
    </div>
  );
}

/** The compact strip in the body panel: pose on the left, skin tone on the right. */
export function BodyStrip() {
  return (
    <div className="body-strip" role="group" aria-label="Body">
      <PosePicker />
      <div className="menu-field"><span>Skin tone</span><SkinTonePicker /></div>
    </div>
  );
}
