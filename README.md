# musical-marble-generator

Gerador headless de vídeos musicais 3D em 9:16, com cinemática determinística
dirigida por MIDI. Não usa Rigid Body.

Requer Python 3.10+, Blender 4.0.2 com Cycles CUDA e suporte H.264, FFmpeg, a biblioteca
nativa FluidSynth e uma SoundFont General MIDI `.sf2` fornecida pelo usuário.
No Debian/Ubuntu, os pacotes `blender`, `ffmpeg`, `libfluidsynth3` e
`fluid-soundfont-gm` fornecem as ferramentas e uma SoundFont; confirme que a
versão instalada do Blender é 4.0.2, como no ambiente Google Colab deste projeto.
No Colab, selecione um runtime com GPU NVIDIA. O builder exige CUDA, ativa somente
as GPUs CUDA nas preferências do Cycles e define `scene.cycles.device = 'GPU'`.
Se CUDA ou a GPU não estiverem disponíveis, o script encerra com erro antes de
construir a geometria. Não há fallback para CPU. O máximo é de 64 samples por
frame; `--render-samples` permite menos amostras para prévias e valores maiores
são limitados a 64. O vidro mantém a aproximação translúcida por shader.
O denoising de render, prévia e camadas é desativado explicitamente, assim como
os passes auxiliares de denoising e o compositor, para suportar builds sem
OpenImageDenoise. Não é necessário instalar nem selecionar um denoiser.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py input.mid --soundfont /caminho/GeneralUser.sf2 -o output_final.mp4
```

Também é possível definir `SOUNDFONT=/caminho/GeneralUser.sf2` e executar
`python src/main.py input.mid`. Os pacotes Python são instalados no Python do CLI;
o Python embutido no Blender usa apenas `bpy` e a biblioteca padrão.

O CLI interpreta mudanças de andamento, agrupa notas simultâneas em um contato
da bolinha e calcula os arcos com o solver. Em seguida, entrega um JSON ao Blender,
renderiza frames PNG a 1080×1920 e 60 FPS, codifica H.264 pelo FFmpeg e sintetiza WAV mono PCM de 24 bits a
48 kHz com FluidSynth e combina vídeo e áudio AAC de 320 kb/s pelo FFmpeg.
Sem limite de duração, todas as notas e instrumentos do MIDI original permanecem no áudio.

Os tempos são absolutos: a nota em `t` corresponde ao frame `1 + 60*t`.
Contatos entre frames inteiros mantêm chaves fracionárias. O silêncio inicial é
preservado e a câmera/bolinha aguardam no primeiro contato; após o último contato,
a imagem permanece na última plataforma até o fim do MIDI mais um segundo de
cauda. A duração é arredondada para frames completos e aplicada também ao áudio.
Os frames e o plano da cena permanecem no cache após sucesso ou falha; apenas os
temporários de codificação e áudio são removidos. A saída só é publicada após o
muxing. Use `--overwrite` para substituir um vídeo existente.

Para persistir o progresso no Google Drive, monte-o no notebook do Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Depois execute:

```bash
python src/main.py input.mid --soundfont instrumento.sf2 --max-duration 15 \
  --cache-dir "/content/drive/MyDrive/marble-cache" \
  -o "/content/drive/MyDrive/output_final.mp4"
```

`--cache-dir` usa uma pasta já montada; sem a opção, o cache fica em
`.marble-cache` ao lado da saída. Cada render tem uma subpasta identificada pelo
conteúdo do MIDI, plano, configurações e código do builder (incluindo câmera e
materiais). Mudanças nesses dados geram outro cache automaticamente.

Os arquivos são `frames/frame_000001.png`, `frame_000002.png`, etc. O Blender
grava primeiro um `.png.part` e só publica o PNG após verificar sua integridade.
O CLI só pula a etapa 3D se houver **exatamente** os frames `1..N` esperados, sem
PNGs extras, todos com dimensões corretas, CRCs e dados comprimidos válidos.
Sequências incompletas retomam apenas frames ausentes ou corrompidos. Uma
desconexão não apaga os frames concluídos na pasta persistente. Na próxima sessão,
monte o mesmo Drive e use o mesmo comando e diretório de cache.

`--force` apaga a sequência do cache daquele render e começa do zero, mesmo que
esteja completa. Outros renders na raiz do cache são preservados. Se o MP4 final
já existir, acrescente também `--overwrite`; essa opção controla apenas a saída.

Para uma prévia rápida, reduza resolução e amostras:

```bash
python src/main.py input.mid --soundfont instrumento.sf2 --resolution-percentage 10 --render-samples 1
python src/main.py --help
```

Para limitar uma música longa aos primeiros 15 segundos:

```bash
python src/main.py input.mid --soundfont instrumento.sf2 --max-duration 15
```

O limite é absoluto, contado desde o início do MIDI, incluindo o silêncio inicial
e a cauda sonora. O solver descarta notas posteriores antes de calcular os arcos;
o JSON contém apenas os contatos e segmentos do trecho. O Blender cria somente
suas plataformas/pinos e dimensiona a parede e a animação para esse trecho.
O áudio também é filtrado antes da síntese. A duração final nunca excede o limite:
é arredondada para baixo em frames completos a 60 FPS. Notas exatamente no limite
continuam na matemática, mas não acrescentam um frame ao vídeo. O trecho precisa
conter pelo menos dois contatos distintos. Sem `--max-duration`, a música inteira
é processada como antes.

`--tail-seconds` ajusta a cauda (acima de 1 s, acrescenta silêncio) e `--descent`
ajusta a descida visual. O uso independente da Fase 2 continua disponível:
`blender --background --python src/blender_builder.py` salva `test_scene.blend`.
O pipeline usa o mesmo builder com `--scene-json` e `--video-output` após `--`.

`src.audio_renderer.render_tts("Piano", "anuncio.mp3")` gera um anúncio separado
via gTTS, que requer internet e envia o texto ao serviço. Essa função não é
chamada pelo pipeline musical.

```bash
python -m unittest discover -s tests -v
# Com SOUNDFONT definido, a suíte também verifica síntese real via FluidSynth.
blender --background --python-exit-code 1 --python tests/blender_40_smoke.py
# O teste de integração acima exige exatamente Blender 4.0.2 e GPU CUDA.
```
