#!/bin/bash
OUTPUT_FILE="$TATBOT_ROOT/output/specs.md"
echo "# Specs for \`$BACKEND\`" > "$OUTPUT_FILE"
echo "Generated on $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
declare -A COMMANDS=(
  ["lsb_release -a"]="🖥️ Operating System Information"
  ["uname -a"]="🐧 Kernel Information"
  ["lspci -vnn | grep VGA"]="🎮 GPU Information"
  ["nvidia-smi"]="🚀 NVIDIA GPU Information"
  ["cat /etc/nv_tegra_release"]="🤖 NVIDIA Jetson Information"
  ["docker version"]="🐋 Docker Version"
  ["docker info"]="🐳 Docker Information"
  ["ip addr"]="🌐 IP Address Information"
  ["ip route"]="📡 Routing Information"
  ["cat /etc/resolv.conf"]="🔍 DNS Resolver Information"
  ["lscpu"]="🧠 CPU Information"
  ["free -h"]="💾 RAM Information"
  ["df -h"]="💽 Disk Information"
)
for CMD in "${!COMMANDS[@]}"; do
  echo -e "\n${COMMANDS[$CMD]}\n\`\`\`" >> "$OUTPUT_FILE"
  eval $CMD >> "$OUTPUT_FILE" 2>&1 || echo "Command failed" >> "$OUTPUT_FILE"
  echo -e "\`\`\`\n" >> "$OUTPUT_FILE"
done
echo "system specs saved to $OUTPUT_FILE"
cat "$OUTPUT_FILE"