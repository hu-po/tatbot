#!/bin/bash
CMD="cd ~/tatbot && git pull"

nodes=(rpi2 rpi1 trossen-ai ojo ook oop)
emojis=("🍇" "🍓" "🦾" "🦎" "🦧" "🦊")

updated=()
failed=()

for i in "${!nodes[@]}"; do
  node=${nodes[$i]}
  emoji=${emojis[$i]}

  # Skip if this is the current node
  if [[ "$node" == "$(hostname)" ]]; then
    continue
  fi

  ssh "$node" "$CMD"
  if [[ $? -eq 0 ]]; then
    updated+=("$emoji $node")
  else
    failed+=("$emoji $node")
  fi
done

if [[ ${#updated[@]} -gt 0 ]]; then
  echo "Successfully updated: ${updated[*]}"
fi
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed to update: ${failed[*]}" >&2
fi