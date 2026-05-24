# redpine-source-proof

En liten feature-idé: visa användare varför Redpine är värt det genom att jämföra med det bästa det öppna webben kan erbjuda, sida vid sida, på begäran. Med metrics som kan visa på relevans etc. 

## Hur det fungerar

Två anrop körs parallellt vid varje fråga. Redpines svar går direkt till användaren. Webb-RAG-resultatet sparas tyst i bakgrunden. Om användaren någonsin undrar "är det här värt det, eller vad hade webben gett mig?" ett klick visar exakt vad de hade fått utan Redpine genom redpines workspace.

```
fråga → Redpine-anrop → svar till användaren
      → webb-skrapning + RAG → sparas → "Varför Redpine?"-panel
```

## Kör det

```bash
git clone https://github.com/FelixJMartin/redpine-source-proof
cd redpine-source-proof
# lägg till din Redpine API-nyckel i .env
open index.html
```

## Varför?
Nya användare vill se att det är värt det för dem, tänker tidigt stadie, man vill vet hur bra dem är. Denna repo chunkar inte alla nätets artikalr men ska man skala featurn går det lätt att importera. 