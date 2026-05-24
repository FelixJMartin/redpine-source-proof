# redpine-source-proof-feature

En liten feature-idé: visa användare varför Redpine är värt det genom att jämföra med det bästa det öppna webben kan erbjuda, sida vid sida, på begäran.

## Hur det fungerar

Två anrop körs parallellt vid varje fråga. Redpines svar går direkt till användaren. Webb-RAG-resultatet sparas tyst i bakgrunden i en Qdrant vektordatabas. Om användaren undrar "hur står denna data mot att scrapa nätet" kan man i workbenchen jämföra Redpines data mot de webbartiklar som en vanlig AI hade hittat, evaluerade mot ett antal ( kan vara hur många man vill ) auto-genererade frågor om samma data.

```
fråga → Redpine-anrop → svar till användaren
      → webb-skrapning → chunkas → embeddar → Qdrant → "Varför Redpine?"-panel
```

Pipelinen använder en förenklad version av CAAR-logiken för att klassificera och score varje fråga, inte den riktiga algoritmen, men strukturen är där om man vill bygga vidare, gick utefter min förståelse av vad som ska evalueras. 

## En vanlig AI utan Redpine söker på webben och får tillbaka artiklar. Algoritmen skrapar dem, chunkar ner dem och sparar i två lager:

1. `web_cache.json` , råa chunks från artiklarna, läsbart format för att se vad vanilla modeller finner för data på nätet. 
2. Qdrant (SQLite lokalt), samma chunks vektoriserade för semantisk sökning


## Resultat

På en fråga om trending companies och sentiment som jag körde genom att använda redpines dara om media samt scrapa nätet och artiklar. 
- Redpine: 8/10 — specifika företag, exakta siffror, realtidsdata
- Webb RAG: 1/10 — "Trump administration och WSJ"

Över 10 auto-genererade följdfrågor baserade på faktisk data:
- Redpine snitt: 6.2/10
- Webb snitt: 0.3/10

Webben klarade knappt en enda specifik följdfråga. I testet användes Groq API mot claude tillsammans me redpine. Kan implemetera flera API:er för fullständig jämförelse.

## Kör det

```bash
git clone https://github.com/FelixJMartin/redpine-source-proof
cd redpine-source-proof
# lägg till dina nycklar i .env
# REDPINE_API_KEY=...
# GROQ_API_KEY=...
pip install requests beautifulsoup4 groq langchain-text-splitters langchain-huggingface qdrant-client sentence-transformers python-dotenv
python main.py
```

## Varför?

Nya användare i tidigt stadie vill veta hur bra Redpine faktiskt är innan de committrar. Den här featuren gör värdet synligt direkt, inte som marknadsföring utan som mätbar skillnad. Repot chunkar ett begränsat antal källor men arkitekturen är byggd för att skala.
