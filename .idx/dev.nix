{ pkgs, ... }: {
  # Use the stable channel
  channel = "stable-23.11";

  # Packages to install in the environment
  packages = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.virtualenv
    pkgs.nodejs_20
    pkgs.docker
    pkgs.docker-compose
    # Libraries for Python wheels
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.glib
  ];

  # Environment variables
  env = {
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      pkgs.glib
    ];
  };

  # Enable Docker
  services.docker.enable = true;

  # IDX specific configuration
  idx = {
    workspace = {
      # Runs when the workspace is first created
      onCreate = {
        install-dependencies = ''
          python3 -m venv venv
          source venv/bin/activate
          pip install --upgrade pip
          pip install -r app/backend/requirements.txt
          # Instalar dependencias extra para análisis
          pip install matplotlib seaborn scikit-learn
        '';
      };
    };
  };
}
