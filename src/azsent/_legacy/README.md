# Emekli modüller

`data_prep.py`, `gold.py`, `splits.py` — v1'de tekilleştirme, gold birleştirme ve
gruplu bölmeyi **her koşuda yeniden** hesaplayan üç aşama. v2'de bu iş veri
hazırlama zamanına taşındı (`tools/repack_dataset.py`) ve koşu tarafında yalnızca
doğrulama kaldı (`azsent/ingest.py`).

Neden emekli edildiler: bu üç aşama koşunun kritik yolundaydı ve her biri sessiz
başarısızlık üretebiliyordu — gold parçalarının başka kaynaktan doldurulması,
alan adlarının eşleşmemesi, karantina mantığının kısmen uygulanması. Bölme artık
dosyanın kendisinde ve denetlenebilir; boru hattı onu üretmiyor, sınıyor.

Referans için burada duruyorlar; içe aktarılmıyorlar.
