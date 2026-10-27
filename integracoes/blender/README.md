# Ponte local do Blender para o NinoX

O complemento cria uma mesa com tampo retangular ou redondo e quatro pés cilíndricos. Ele aceita apenas essa ação estruturada; não executa código recebido do modelo.

## Instalação

1. No Blender, abra **Edit > Preferences > Add-ons > Install from Disk**.
2. Selecione `nino_blender_bridge.py` e ative **NinoX Local Bridge**.
3. Deixe o Blender aberto enquanto o Nino envia o pedido.
4. No projeto, configure `GEMINI_API_KEY` ou use uma instância local compatível para o Nino interpretar o pedido e chamar a ferramenta.

O modo do Nino é salvo em `config/nino_settings.json`. Ele começa como `local`; a fala “mude para modo online” seleciona os provedores de nuvem e “mude para modo local” volta ao processamento local. O modo “automático” tenta primeiro os provedores locais e usa os online se necessário. Uma falha online volta ao modo local e mostra um aviso na tela.

O complemento escuta somente em `127.0.0.1:8765`. Ele gera uma chave local em `%LOCALAPPDATA%\NinoX\blender_token.txt`; o Nino lê essa chave para autenticar o comando. Não compartilhe esse arquivo.

As medidas aceitam metros e centímetros. Se não forem especificadas, o modelo usa 1,20 m de largura, 0,75 m de profundidade, 0,75 m de altura e tampo de 0,04 m. Se você pedir uma mesa redonda, o diâmetro define o tamanho do tampo.
