# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-23.11"; # or "unstable"

  # Use https://search.nixos.org/packages to find packages
  packages = [
    # 1. Herramientas Básicas
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.nodejs_20
    pkgs.docker
    pkgs.docker-compose
    
    # 2. Librerías de Sistema (CRÍTICAS para que pandas/numpy no fallen al compilar)
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.glib
  ];

  # Sets environment variables in the workspace
  env = {
    # Ayuda a que Python encuentre las librerías C++ necesarias
    LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
  };

  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      "ms-python.python"
      "rangav.vscode-thunder-client"
    ];

    # Enable previews
    previews = {
      enable = true;
      previews = {
        # 1. PREVIEW DEL BACKEND (API)
        web = {
          # Entramos a la carpeta correcta antes de ejecutar
          command = ["/bin/bash" "-c" "cd app/backend && python -m uvicorn main:app --host 0.0.0.0 --port $PORT --reload"];
          manager = "web";
        };
      };
    };

    # Workspace lifecycle hooks
    workspace = {
      # Se ejecuta CUANDO SE CREA la máquina (1 sola vez)
      onCreate = {
        # Corregimos las rutas: Entra a app/frontend y app/backend
        npm-install = "cd app/frontend && npm install";
        
        # Creamos entorno virtual DENTRO de app/backend para ser ordenados
        setup-backend = ''
          cd app/backend
          python -m venv .venv
          source .venv/bin/activate
          pip install --upgrade pip
          pip install -r requirements.txt
        '';
      };
      
      # Se ejecuta CADA VEZ que abres el proyecto
      onStart = {
        # Aseguramos que los contenedores de Docker arranquen si usas docker-compose
        # start-docker = "docker-compose up -d";
      };
    };
  };
}
