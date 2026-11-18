; Inno Setup 6 script: installs dist\Mekasa for the current user (no administrator needed).
; Built by scripts/build-release.ps1 with /DMyAppVersion=<version>.
; User data (configuration, calibrations, reports, logs) lives in %LOCALAPPDATA%\Mekasa and is
; kept on uninstall, so a reinstall or update keeps the person's calibrations and vocabulary.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#define MyAppName "Mekasa"

[Setup]
AppId={{6B0E5F1C-5D2A-4E8E-9C61-4A0E8D6F2B71}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Mekasa
DefaultDirName={localappdata}\Programs\Mekasa
DefaultGroupName=Mekasa
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputBaseFilename=Mekasa-{#MyAppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
LicenseFile=..\LICENSE
UninstallDisplayIcon={app}\Mekasa.exe
SourceDir=.

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "Crear un acceso directo en el escritorio"; GroupDescription: "Accesos directos:"

[Files]
Source: "..\dist\Mekasa\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Mekasa (tablero)"; Filename: "{app}\Mekasa.exe"; Parameters: "tablero"
Name: "{group}\Mekasa - prueba de precisión"; Filename: "{app}\Mekasa.exe"; Parameters: "prueba"
Name: "{group}\Mekasa - demostración (con cámara)"; Filename: "{app}\Mekasa.exe"; Parameters: "demo"
Name: "{group}\Mekasa - ver cámaras"; Filename: "{app}\Mekasa-consola.exe"; Parameters: "camaras"
Name: "{group}\Mekasa - datos y vocabulario"; Filename: "{app}\Mekasa.exe"; Parameters: "carpeta"
Name: "{group}\Desinstalar Mekasa"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Mekasa"; Filename: "{app}\Mekasa.exe"; Parameters: "tablero"; Tasks: desktopicon

[Run]
Filename: "{app}\Mekasa.exe"; Parameters: "tablero"; Description: "Abrir Mekasa ahora"; Flags: nowait postinstall skipifsilent
