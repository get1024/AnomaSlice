@echo off
echo Starting build process...
echo Installing PyInstaller...
pip install pyinstaller

echo Cleaning up previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.spec del *.spec

echo Building EXE...
pyinstaller --noconsole --onefile --name "ImageSlicer" src/main.py

echo Build complete!
echo You can find the executable in the 'dist' folder.
pause
