{
  description = "FAR Planner native module — visibility-graph route planner";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/ec7c70d12ce2fc37cb92aff673dcdca89d187bae";
    flake-utils.url = "github:numtide/flake-utils";
    dimos-lcm = {
      url = "github:dimensionalOS/dimos-lcm/main";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, dimos-lcm, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Override LCM to skip logger (fdatasync not on macOS)
        lcm-patched = pkgs.lcm.overrideAttrs (old: {
          cmakeFlags = (old.cmakeFlags or []) ++ [
            "-DLCM_ENABLE_TESTS=OFF"
            "-DLCM_INSTALL_EXAMPLES=OFF"
          ];
          # Patch fdatasync for macOS
          postPatch = (old.postPatch or "") + ''
            if [ "$(uname)" = "Darwin" ]; then
              sed -i 's/fdatasync/fsync/g' lcm-logger/lcm_logger.c || true
            fi
          '';
        });

        # PCL without VTK (avoids tiledb/pdal build failures)
        pcl-minimal = pkgs.pcl.override {
          vtk = null;
        };

        smart-nav-common = ./common;

        far_planner_native = pkgs.stdenv.mkDerivation {
          pname = "far_planner_native";
          version = "0.1.0";

          src = ./.;

          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs = [
            lcm-patched
            pkgs.glib
            pkgs.eigen
            pcl-minimal
            pkgs.opencv
            pkgs.boost
            pkgs.llvmPackages.openmp
          ];

          cmakeFlags = [
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
            "-DFETCHCONTENT_SOURCE_DIR_DIMOS_LCM=${dimos-lcm}"
            "-DSMART_NAV_COMMON_DIR=${smart-nav-common}"
          ];
        };
      in {
        packages = {
          default = far_planner_native;
          inherit far_planner_native;
        };
      });
}
