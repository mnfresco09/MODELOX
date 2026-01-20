{ pkgs, ... }: {
  # Let Nix manage pkgs
  environment.systemPackages = [
    pkgs.python3
    pkgs.nodejs_20
    pkgs.docker
    pkgs.docker-compose
  ];

  # Set environment variables
  environment.variables = {
    # Example:
    # "MY_VARIABLE" = "my-value";
  };

  # Start services
  services.docker.enable = true;


  # Custom commands
  # See https://developers.google.com/idx/guides/customize-idx-env#nix-commands
  # for more details
  idx.commands = [
    {
      name = "Setup and Install Backend Dependencies";
      command = '''
        if [ ! -d "venv" ]; then
          echo "Creating Python virtual environment..."
          python3 -m venv venv
        fi
        echo "Installing backend dependencies..."
        ./venv/bin/pip install -r app/backend/requirements.txt
      ''';
    }
    {
      name = "Install Frontend Dependencies";
      command = "cd app/frontend && npm install";
    }
  ];
}