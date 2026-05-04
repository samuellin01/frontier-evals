#!/bin/bash
set -e

echo "=== PaperBench Setup Script ==="

# 1. Install podman and socat if not available
if ! command -v podman &>/dev/null; then
    echo "Installing podman..."
    sudo dnf install -y podman
fi
if ! command -v socat &>/dev/null; then
    echo "Installing socat..."
    sudo dnf install -y socat
fi
echo "Podman version: $(podman --version)"

# 2. Start podman socket service for Docker API compatibility
PODMAN_SOCKET="unix:///run/user/$(id -u)/podman/podman.sock"
echo "Starting podman socket service..."
systemctl --user enable --now podman.socket 2>/dev/null || true

export DOCKER_HOST="$PODMAN_SOCKET"
echo "DOCKER_HOST=$DOCKER_HOST"

# 3. Start socat relay for container internet access
# The corporate proxy (fwdproxy) rejects connections from inside rootless
# containers. socat relays container traffic through the host process.
if ! ss -tlnp6 | grep -q ":18080 " 2>/dev/null; then
    echo "Starting socat proxy relay on port 18080..."
    socat TCP6-LISTEN:18080,fork,reuseaddr TCP6:fwdproxy:8080 &
    sleep 1
fi

# 4. Fetch Git LFS data
echo "Fetching Git LFS data..."
cd /home/samuellin/workspace/frontier-evals
git lfs fetch --include "project/paperbench/data/**" --exclude ""
git lfs checkout project/paperbench/data
cd /home/samuellin/workspace/frontier-evals/project/paperbench

# 5. Set up environment variables
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << 'ENVEOF'
OPENAI_API_KEY=dummy-for-testing
GRADER_OPENAI_API_KEY=dummy-for-testing
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
OPENROUTER_API_KEY=
ENVEOF
fi
source .env

if [ ! -f paperbench/solvers/agent.env ]; then
    echo "Creating agent.env..."
    cat > paperbench/solvers/agent.env << 'ENVEOF'
OPENAI_API_KEY=dummy-for-testing
HF_TOKEN=
ENVEOF
fi

# 6. Build container images with podman
# Use socat relay (port 18080) as proxy for container builds
echo "Building container images with podman..."
cd /home/samuellin/workspace/frontier-evals/project/paperbench
podman build --network=host \
    --build-arg http_proxy="http://[::1]:18080" \
    --build-arg https_proxy="http://[::1]:18080" \
    --build-arg HTTP_PROXY="http://[::1]:18080" \
    --build-arg HTTPS_PROXY="http://[::1]:18080" \
    --platform=linux/amd64 -t pb-env -f paperbench/Dockerfile.base .
podman build --network=host \
    --build-arg http_proxy="http://[::1]:18080" \
    --build-arg https_proxy="http://[::1]:18080" \
    --build-arg HTTP_PROXY="http://[::1]:18080" \
    --build-arg HTTPS_PROXY="http://[::1]:18080" \
    --platform=linux/amd64 -t pb-reproducer -f paperbench/reproducer.Dockerfile .

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Add the following to your shell profile:"
echo "  export DOCKER_HOST=$PODMAN_SOCKET"
echo "  export PAPERBENCH_DATA_DIR=/home/samuellin/workspace/frontier-evals/project/paperbench/data"
echo ""
echo "To run the e2e dummy test:"
echo "  source .env"
echo "  /home/samuellin/.conda/envs/paperbench/bin/python -m paperbench.nano.entrypoint \\"
echo '    paperbench.paper_split=debug \'
echo '    paperbench.solver=paperbench.solvers.dummy.solver:PaperBenchDummySolver \'
echo '    "paperbench.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntimeNoJupyter" \'
echo '    "paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig" \'
echo '    paperbench.solver.computer_runtime.env.pull_from_registry=false \'
echo '    paperbench.solver.computer_runtime.env.local_network=true \'
echo '    paperbench.reproduction.skip_reproduction=true \'
echo '    paperbench.judge.scaffold=dummy \'
echo '    runner.recorder=nanoeval.json_recorder:json_recorder'
