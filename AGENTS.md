# Regras de execução para agentes

Estas regras valem para todo o repositório. Não as flexibilize por iniciativa
própria; mudanças de escopo ou das restrições exigem instrução expressa do usuário.

## Uma tarefa por vez

- Leia `PLANO.md` e a solicitação atual antes de editar arquivos.
- Execute apenas o passo explicitamente autorizado e apenas nos arquivos indicados.
- Entregue um resultado revisável e aguarde aprovação antes de iniciar outro passo.
- Não acrescente melhorias, refatorações, automações ou dependências fora do escopo.

## Física e áudio intocáveis

- Não altere `src/midi_parser.py`, `src/layout_solver.py` nem
  `src/audio_renderer.py`, nem a matemática de timestamps, impactos, trajetórias,
  FPS ou alinhamento entre áudio e vídeo, durante etapas visuais/organizacionais.
- É permitido ler os dados de colisão existentes para sincronizar efeitos visuais.
- A arquitetura usa cinemática determinística; não introduza Rigid Body.

## Validação limitada

- Validações visuais devem usar `--test-frame N`: somente um frame novo, sem
  reaproveitar o cache da animação. Nunca renderize os 900 frames como teste.
- Não renderize músicas ou vídeos inteiros para testar iluminação ou materiais.
- Um draft curto, como `--draft --seconds 2`, só pode ser executado quando a
  etapa autorizada exigir explicitamente esse teste de pipeline.
- Em tarefas documentais, não execute renderizações nem a suíte de código;
  verifique apenas Markdown, padrões de ignore e o escopo dos arquivos alterados.
- Não afirme ter validado CUDA, denoising ou uma versão do Blender sem essa
  execução real; informe as limitações do ambiente quando forem relevantes.

## Artefatos e escopo do commit

- Use `output/` para previews, vídeos, áudios e caches gerados. Esse é um destino
  recomendado pelos comandos da documentação; não altere defaults do código em
  uma tarefa documental.
- Não versione frames, PNGs, MP4s, MIDI de teste, SoundFonts, `.blend` ou caches.
- Preserve `output/.gitkeep`. Não apague resultados do usuário para limpar o Git.
- Use listas explícitas de arquivos ao preparar commits; não use `git add .` ou
  `git add -A`. Não contorne o `.gitignore` para incluir artefatos pesados.
- Antes de publicar, confirme que somente os arquivos autorizados estão no diff.

## Checklist, commits e tags

- Cada passo entregue deve ter um commit descritivo, contendo apenas seu escopo.
- Em `PLANO.md`, use `[ ] Pendente` e `[x] Concluído no commit XYZ`; respeite os
  estados e referências de commit definidos pelo usuário. Não marque passos
  futuros como concluídos nem interprete a entrega como aprovação automática.
- Ao concluir um passo, crie e publique uma Git Tag apontando para o commit
  correspondente. Use o nome solicitado pelo usuário; não mova tags existentes.
- Confirme o SHA remoto do commit e da tag antes de declarar a publicação.
- Para este marco organizacional, a tag solicitada é `v0.5-org`.
