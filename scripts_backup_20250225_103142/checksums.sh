mkdir -p "${PACKAGE_DIR}/checksums"
for pkg in "${PACKAGE_DIR}/wheels"/*.whl; do
    sha256sum "$pkg" > "${PACKAGE_DIR}/checksums/$(basename ${pkg%.*}).sha256"
done 