# PLANO.md - Musical Marble Generator
## Fase 2: Evolução para Fotorrealismo (Blender/Cycles)

### Regras de Execução
1. O Codex deve executar apenas UM passo por vez e aguardar aprovação.
2. Teste de 1 Frame: Para validar cada passo visual, o código deve ser capaz de renderizar um único frame de teste (ex: frame 150, onde ocorre um impacto), ignorando ou sobrescrevendo o cache antigo. Nunca renderizar os 900 frames durante os testes.
3. Leitura Permitida: O core da física e sincronização de áudio não deve ser alterado, mas o Codex pode (e deve) ler os dados de colisão gerados para sincronizar os materiais visuais.

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
