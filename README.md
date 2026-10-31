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
renderiza MP4/H.264 a 1080×1920 e 60 FPS, sintetiza WAV mono PCM de 24 bits a
48 kHz com FluidSynth e combina vídeo e áudio AAC de 320 kb/s pelo FFmpeg.
Sem limite de duração, todas as notas e instrumentos do MIDI original permanecem no áudio.

Os tempos são absolutos: a nota em `t` corresponde ao frame `1 + 60*t`.
Contatos entre frames inteiros mantêm chaves fracionárias. O silêncio inicial é
preservado e a câmera/bolinha aguardam no primeiro contato; após o último contato,
a imagem permanece na última plataforma até o fim do MIDI mais um segundo de
cauda. A duração é arredondada para frames completos e aplicada também ao áudio.
O diretório temporário é removido ao concluir ou falhar, e a saída só é publicada
após o muxing. Use `--overwrite` para substituir um vídeo existente.

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
