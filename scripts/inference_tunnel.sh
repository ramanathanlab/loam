#!/usr/bin/env bash
# Keep in the foreground; stop with Ctrl-C. No port defaults are assumed.
set -euo pipefail
usage() {
    echo 'Laptop -> fixed Jupiter login: inference_tunnel.sh reverse LOGIN_SSH_HOST LOGIN_ARGO_PORT LOCAL_ARGO_PORT' >&2
    echo 'Allocated compute -> login: inference_tunnel.sh compute LOGIN_SSH_HOST COMPUTE_ARGO_PORT LOGIN_ARGO_PORT' >&2
    exit 2
}
[[ $# == 4 ]] || usage
MODE=$1
LOGIN=$2
for value in "$3" "$4"; do
    [[ "$value" =~ ^[0-9]+$ && ${#value} -le 5 ]] || usage
    (( 10#$value >= 1 && 10#$value <= 65535 )) || usage
done
case "$MODE" in
    reverse)
        exec ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
            -R "127.0.0.1:$3:127.0.0.1:$4" "$LOGIN"
        ;;
    compute)
        exec ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
            -L "127.0.0.1:$3:127.0.0.1:$4" "$LOGIN"
        ;;
    *) usage ;;
esac
