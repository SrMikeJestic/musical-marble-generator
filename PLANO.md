# PLANO.md - Musical Marble Generator
## Fase 2: Evolução para Fotorrealismo (Blender/Cycles)

### Regras de Execução
1. O Codex deve executar apenas UM passo por vez e aguardar aprovação.
2. Teste de 1 Frame: Para validar cada passo visual, o código deve ser capaz de renderizar um único frame de teste (ex: frame 150, onde ocorre um impacto), ignorando ou sobrescrevendo o cache antigo. Nunca renderizar os 900 frames durante os testes.
3. Leitura Permitida: O core da física e sincronização de áudio não deve ser alterado, mas o Codex pode (e deve) ler os dados de colisão gerados para sincronizar os materiais visuais.

### Passo 0: Otimização de Pipeline e Gestão de Cache (Kaggle)
*   **Objetivo:** Preparar a infraestrutura do orquestrador com modos de teste eficientes, validação de cache inteligente e suporte ao ambiente Kaggle.
*   **Ação:**
    1. Implementar o argumento `--seconds N` (calcula e renderiza apenas os primeiros N segundos da animação).
    2. Implementar o modo `--draft` (configura resolução reduzida, amostras/samples baixos, ativa *denoising* e executa o pipeline completo gerando o vídeo com áudio MIDI integrado).
    3. Isolar os diretórios de cache: criar lógica para separar fisicamente os frames de `--test-frame`, do modo `--draft` e do render final de alta qualidade.
    4. Validação rigorosa do cache: o script só deve pular a etapa de renderização 3D se a contagem de imagens na pasta de cache for exatamente igual ao número de frames calculados para a duração solicitada.
    5. Implementar flag `--force` para ignorar a validação e forçar a reescrita do cache.
    6. Fornecer a célula de código exata (com `apt-get` e `pip`) para instalar as dependências de áudio no ambiente de um notebook Kaggle limpo.
*   **Critério de Conclusão:** Execução de um comando `--draft --seconds 2` gera um vídeo rápido com áudio, isolado na sua própria pasta de cache, validando que frames faltantes ou a presença do `--force` reativam o Blender corretamente.

### Passo 1: Iluminação Dramática e Fundo
*   **Objetivo:** Eliminar o aspeto "chapado" e criar uma iluminação de estúdio contrastante.
*   **Ação:** Configurar o `World` com cor escura. Adicionar luz principal do tipo `Spot` ou `Area` focada na trajetória.
*   **Critério de Conclusão:** Render de 1 frame específico gerado ignorando o cache, exibindo alto contraste de luz e sombra.

### Passo 2: Materiais Avançados (Shaders) e Otimização
*   **Objetivo:** Materiais PBR (metal e vidro) com controle de ruído/tempo.
*   **Ação:** Metal polido na esfera, vidro translúcido (`Glass BSDF`) nas teclas. Ler os dados de impacto para animar a propriedade `Emission` dos blocos. Configurar *samples* de render baixos (ex: 32) para rascunho.
*   **Critério de Conclusão:** Render de 1 frame de impacto comprovando os reflexos e brilho neon sem estourar o tempo de render.

### Passo 3: Óptica Cinematográfica (Depth of Field)
*   **Objetivo:** Simular câmera real com desfoque de fundo.
*   **Ação:** Ativar `Depth of Field` (DoF) focado na esfera (F-stop 1.8 a 2.8).
*   **Critério de Conclusão:** Render de 1 frame validando o desfoque em profundidade.

### Passo 4: Estrutura Física (Carris/Suportes)
*   **Objetivo:** Dar suporte físico aos blocos.
*   **Ação:** Adicionar cilindros/carris de metal unindo a trajetória.
*   **Critério de Conclusão:** Cenário com base estrutural contínua.
