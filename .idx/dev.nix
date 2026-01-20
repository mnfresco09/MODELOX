{ pkgs, ... }: {
  # Let Nix manage pkgs
  environment.systemPackages = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.virtualenv
    pkgs.nodejs_20
    pkgs.docker
    pkgs.docker-compose
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];

  # Set environment variables
  environment.variables = {
    # Fix: Asegura que los paquetes instalados con pip encuentren las librerias de C++ necesarias
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
    ];
  };

  # Start services
  services.docker.enable = true;

  # Custom commands
  idx.commands = [
    {
      name = "Setup and Install Backend Dependencies";
      command = ''
        if [ ! -d "venv" ]; then
          echo "Creating Python virtual environment..."
          python3 -m venv venv
        fi
        echo "Installing backend dependencies..."
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r app/backend/requirements.txt
      '';
    }
    {
      name = "Install Frontend Dependencies";
      command = "cd app/frontend && npm install";
    }
  ];
}