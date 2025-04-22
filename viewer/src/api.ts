export async function saveTransforms(data: Record<string, number[]>) {
    const r = await fetch("/save-xforms", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    if (!r.ok) throw new Error(await r.text());
  }