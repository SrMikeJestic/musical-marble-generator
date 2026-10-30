# musical-marble-generator

Gerador headless de vídeos musicais 3D em 9:16, com cinemática determinística
dirigida por MIDI. Não usa Rigid Body.

Requer Python 3.10+, Blender 4.2+ com EEVEE e suporte H.264, FFmpeg, a biblioteca
nativa FluidSynth e uma SoundFont General MIDI `.sf2` fornecida pelo usuário.
No Debian/Ubuntu, os pacotes `blender`, `ffmpeg`, `libfluidsynth3` e
`fluid-soundfont-gm` fornecem as ferramentas e uma SoundFont; confirme que a
versão instalada do Blender é pelo menos 4.2.

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
Todas as notas e instrumentos do MIDI original permanecem no áudio.

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
```
