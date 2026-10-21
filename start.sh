#!/bin/bash
# 🐋 WhaleTracker - Hızlı Başlangıç (Docker gerektirmez)

echo "🐋 WhaleTracker kuruluyor..."

# 1. Sanal ortam oluştur
if [ ! -d "venv" ]; then
    echo "📦 Sanal ortam oluşturuluyor..."
    python -m venv venv
fi

# 2. Sanal ortamı aktifleştir
source venv/bin/activate

# 3. Bağımlılıkları kur
echo "📥 Bağımlılıklar kuruluyor..."
pip install -r requirements.txt -q

# 4. .env oluştur (yoksa)
if [ ! -f ".env" ]; then
    echo "⚙️ .env dosyası oluşturuluyor..."
    cp .env.example .env
fi

# 5. Logs klasörü
mkdir -p logs

echo ""
echo "✅ Kurulum tamamlandı!"
echo ""
echo "🚀 Başlatmak için:"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
echo "   Dashboard: http://localhost:8000"
echo ""
