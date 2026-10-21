#!/usr/bin/env python3
"""
🐋 WhaleTracker - Tek tıkla başlatma
Docker gerektirmez, sadece Python gerekli.

Kullanım:
    python start.py              # Varsayılan ayarlarla başlat
    python start.py --port 3000  # Özel port
    python start.py --debug      # Debug modu
"""

import subprocess
import sys
import os
from pathlib import Path


def check_python():
    """Python sürümünü kontrol et."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ gerekli!")
        print(f"   Mevcut: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")


def install_deps():
    """Bağımlılıkları kur."""
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print("❌ requirements.txt bulunamadı!")
        sys.exit(1)

    print("📥 Bağımlıklar kuruluyor...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "-r", str(req_file), "-q"
    ])
    print("✅ Bağımlılıklar kuruldu")


def create_env():
    """Yoksa .env oluştur."""
    env_file = Path(__file__).parent / ".env"
    env_example = Path(__file__).parent / ".env.example"

    if not env_file.exists() and env_example.exists():
        print("⚙️  .env dosyası oluşturuluyor...")
        env_file.write_text(env_example.read_text())
        print("✅ .env oluşturuldu")


def create_dirs():
    """Gerekli klasörleri oluştur."""
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)


def main():
    print("=" * 50)
    print("🐋 WhaleTracker Başlatıcı")
    print("=" * 50)
    print()

    # Kontroller
    check_python()
    create_env()
    create_dirs()

    # Bağımlılık kontrolü
    try:
        import yfinance
        print("✅ yfinance yüklü")
    except ImportError:
        install_deps()

    # main.py'yi çalıştır
    print()
    print("🚀 WhaleTracker başlatılıyor...")
    print("   Dashboard: http://localhost:8000")
    print("   Durdurmak için: Ctrl+C")
    print()

    # Argümanları ilet
    sys.argv = [sys.argv[0]]  # Temizle
    if len(sys.argv) > 1:
        sys.argv.extend(sys.argv[1:])

    # main.py'yi import et ve çalıştır
    os.chdir(Path(__file__).parent)
    from main import main as run_main
    run_main()


if __name__ == "__main__":
    main()
