# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-23.11"; # or "unstable"

  # Use https://search.nixos.org/packages to find packages
  packages = [
    # Core
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.nodejs_20
    pkgs.nodePackages.npm

    # System utilities for psutil and others
    pkgs.gcc
    pkgs.gnumake
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];

  # Sets environment variables in the workspace
  env = {
    # Fix for python packages sometimes needing this
    LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
  };

  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      "ms-python.python"
      "ms-python.debugpy"
      "esbenp.prettier-vscode"
      "bradlc.vscode-tailwindcss"
    ];

    # Enable previews
    previews = {
      enable = true;
      previews = {
        # The web preview
        web = {
          # Example: run "npm run dev" with PORT set to IDX's defined port for previews,
          # and show it in the web preview panel
          command = ["npm" "run" "dev" "--" "--port" "$PORT" "--host" "0.0.0.0"];
          manager = "web";
          cwd = "frontend";
        };
      };
    };

    # Workspace lifecycle hooks
    workspace = {
      # Runs when a workspace is first created
      onCreate = {
        # Backend setup - Installing ALL dependencies
        install-backend = "pip install -r backend/requirements.txt";
        
        # Frontend setup
        install-frontend = "cd frontend && npm install";
      };
      
      # Runs when the workspace is (re)started
      onStart = {
        # Run backend in background
        start-backend = "cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload &";
      };
    };
  };
}
