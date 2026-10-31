# Musical Marble Generator

Gerador headless de vídeos musicais 3D em 9:16, com cinemática determinística
orientada por MIDI. O projeto usa Blender 4.0.2, Cycles CUDA e FFmpeg. Não usa
Rigid Body. As regras de contribuição estão em [AGENTS.md](AGENTS.md) e o
checklist de passos está em [PLANO.md](PLANO.md).

## Como rodar no Kaggle: três células

Antes de começar, habilite **Internet** e uma **GPU NVIDIA** nas configurações do
notebook. Adicione um dataset com seu MIDI e substitua o caminho de exemplo na
célula 3. Execute as células na ordem. Estas instruções são documentação: nenhum
script de automação ou default do código é alterado por este marco.

### Célula 1 — Instalar Blender 4.0.2 e dependências de áudio

O Blender é baixado do distribuidor oficial com versão fixa. A SoundFont General
MIDI de teste é fornecida pelo pacote `timgm6mb-soundfont`. Os pacotes Python
abaixo correspondem ao `requirements.txt` e são instalados no Python do notebook,
não no Python embutido do Blender.

```python
!apt-get update -qq
!DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git wget xz-utils ffmpeg libfluidsynth3 libsndfile1 timgm6mb-soundfont libxi6 libxrender1 libxkbcommon0 libsm6 libgl1
!wget -q "https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz" -O "/kaggle/working/blender-4.0.2-linux-x64.tar.xz"
!tar -xf "/kaggle/working/blender-4.0.2-linux-x64.tar.xz" -C "/kaggle/working"
%pip install -q "pretty_midi>=0.2.11,<0.3" "pyfluidsynth>=1.3.3,<2" "numpy>=1.26,<3" "soundfile>=0.12,<1" "gTTS>=2.5,<3"

import os
os.environ["PATH"] = "/kaggle/working/blender-4.0.2-linux-x64:" + os.environ["PATH"]
os.environ["SOUNDFONT"] = "/usr/share/sounds/sf2/TimGM6mb.sf2"
!blender --version
!nvidia-smi
```

### Célula 2 — Clonar ou atualizar o repositório

A atualização usa apenas fast-forward; não descarta alterações locais.

```python
from pathlib import Path
import subprocess

REPO = Path("/kaggle/working/musical-marble-generator")
if not (REPO / ".git").exists():
    subprocess.run([
        "git", "clone",
        "https://github.com/SrMikeJestic/musical-marble-generator.git", str(REPO)
    ], check=True)
else:
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=True)

%cd /kaggle/working/musical-marble-generator
```

### Célula 3 — Executar um único modo de teste

Use `MODO = "test-frame"` para validar somente o frame 150. Para o teste de
pipeline autorizado no Passo 0, troque para `MODO = "draft"`: serão processados
apenas os primeiros dois segundos, com vídeo e áudio. Não execute vídeos
inteiros como teste. O frame 150 é um exemplo: escolha um frame de impacto do seu
MIDI e confirme que ele cabe no trecho solicitado.

```python
import sys
from IPython.display import Image, Video, display

MODO = "test-frame"  # Ou "draft", somente quando esse teste estiver autorizado.
MIDI = Path("/kaggle/input/seu-dataset/musica.mid")  # Ajuste este caminho.
FORCAR = False      # True refaz os frames do draft, ignorando o cache.
Path("output").mkdir(exist_ok=True)

comando = [sys.executable, "src/main.py", str(MIDI),
           "--cache-dir", "output/marble-cache"]
if MODO == "test-frame":
    comando += ["--test-frame", "150", "--max-duration", "15",
                "--resolution-percentage", "50",
                "--test-output", "output/passo1_frame150.png"]
elif MODO == "draft":
    comando += ["--draft", "--seconds", "2", "--soundfont", os.environ["SOUNDFONT"],
                "-o", "output/draft_2s.mp4", "--overwrite"]
    if FORCAR:
        comando.append("--force")
else:
    raise ValueError("Escolha apenas test-frame ou draft.")

subprocess.run(comando, check=True)
if MODO == "test-frame":
    display(Image(filename="output/passo1_frame150.png"))
else:
    display(Video("output/draft_2s.mp4", embed=True))
```

## Modos e cache

| Modo | Resultado | Cache dentro de `output/marble-cache/` |
| --- | --- | --- |
| `--test-frame N` | Um PNG novo; não usa SoundFont nem FFmpeg | `test-frame/frame_<n>/` |
| `--draft --seconds 2` | MP4 com áudio, até 270×480 e 16 samples, com denoising | `draft/<chave>/frames/` |
| Final, sem `--draft` | Vídeo com áudio, até 1080×1920 e 64 samples | `final/<chave>/frames/` |

`--seconds` é alias de `--max-duration`. Os argumentos `--test-frame`,
`--max-duration`, `--resolution-percentage` e `--test-output` continuam válidos.
O modo de um frame sempre ignora os frames antigos e substitui seu PNG de teste;
`--draft` e `--test-frame` não devem ser combinados.

O CLI só pula o Blender quando há exatamente a sequência de frames esperada,
sem PNGs extras ou inválidos. Verifica dimensões, CRCs e os dados comprimidos dos
PNGs. Frames concluídos são publicados por renomeação de `.png.part`; falhas ou
interrupções preservam os frames válidos para retomada. A chave inclui os bytes
do MIDI, o plano e o código da cena, inclusive câmera e materiais.

`--force` refaz a sequência daquele job, preservando os outros caches. `--overwrite`
autoriza substituir o MP4 de saída; são opções distintas. O limite temporal inclui
a cauda sonora, preserva o silêncio inicial e não ultrapassa o número de segundos
solicitado. São necessários pelo menos dois contatos distintos no trecho.

O render exige CUDA e não faz fallback para CPU. Draft seleciona OptiX quando
disponível e só usa OpenImageDenoise se o build o inclui; sem nenhum denoiser
suportado, mostra um erro. Render final e teste de um frame mantêm denoising
desativado. O compositor permanece desativado em todos os modos.

## Destino seguro e governança

Use `output/` para imagens, vídeos, áudio e caches. Essa convenção dos exemplos
não muda os defaults existentes do CLI. Somente `output/.gitkeep` é versionado;
seu conteúdo e os arquivos pesados de entrada/saída são bloqueados pelo
`.gitignore`, inclusive frames, PNGs, MP4s, MIDI de teste e `marble-cache/`.
Exporte ou salve os outputs do Kaggle para preservar resultados entre sessões.

Cada passo autorizado deve ser entregue em um commit separado, com uma Git Tag
própria, e aguardar aprovação antes do seguinte. O checklist usa `[ ] Pendente` e
`[x] Concluído no commit XYZ`, conforme o estado definido pelo usuário. O marco
de organização do Passo 0.5 recebe a tag `v0.5-org`. Não sobrescreva tags existentes.

A física, o parser MIDI e a sincronização de áudio são somente leitura durante
etapas visuais ou documentais. Este marco altera apenas organização e documentação:
não executa renders, testes de código ou mudanças em arquivos `.py`.
