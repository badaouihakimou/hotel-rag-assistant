# Assistant virtuel d'hôtel : un RAG construit, évalué et mis à l'épreuve



> Un hôtel veut un assistant qui réponde aux questions de ses clients à partir de sa
> documentation interne cinq PDF sans jamais inventer.

RAG écrit à la main : ni LangChain, ni LlamaIndex, ni base vectorielle externe. Extraction,
embeddings, recherche et génération assemblés brique par brique, puis évalués.

## La démarche

| Partie | Contenu |
|---|---|
| 1 | Un LLM seul hallucine : il invente un hôtel 5 étoiles à Paris |
| 2 | Toute la documentation dans le prompt : mieux, mais lent et imprécis |
| 3 | RAG : embeddings, recherche sémantique, réponses sourcées |
| 4 | Évaluation chiffrée et exploration systématique des limites |
| 5 | Application Streamlit |

Les parties 1 à 3 suivent l'exercice d'origine. Les parties 4 et 5 sont un prolongement
personnel.

## Pourquoi une partie 4

Le RAG de l'exercice fonctionne sur quatre questions. C'est une démonstration, pas une
évaluation.

Un RAG peut échouer à deux endroits distincts, et les confondre mène à de mauvaises
corrections :

- la recherche, la bonne rubrique a-t-elle été retrouvée ? Si non, aucun LLM ne rattrapera ;
- la génération, la rubrique était là, le modèle a-t-il bien répondu ?

Changer de LLM ne sert à rien si c'est la recherche qui rate. La partie 4 les mesure séparément.

### Ce qu'elle contient

Un jeu d'évaluation de 24 questions écrites comme un client les poserait : synonymes
(« chiens » pour « animaux »), fautes de frappe, langue étrangère, formulations familières,
et six questions hors sujet.

Trois métriques de recherche : hit@1, hit@2, MRR avec le détail des questions mal
classées, plus instructif que la moyenne.

Un détecteur de questions hors sujet, La distribution des scores sépare les questions du
périmètre de celles qui n'en sont pas ; un seuil calibré dessus permet de refuser sans
appeler le LLM : refus garanti, réponse instantanée, zéro risque d'hallucination.

Une étude du `top_k` : la couverture plafonne pendant que la taille du contexte croît
linéairement. Plus de rubriques n'est pas mieux — c'est exactement le problème de la partie 2.

Une comparaison sémantique / hybride. Les embeddings comprennent le sens, TF-IDF trouve les
mots exacts. Le notebook teste plusieurs pondérations et retient la meilleure, avec préférence
pour la plus simple à égalité.

Sept pièges visant chacun une faiblesse précise : synonyme, question multi-rubriques,
question ambiguë, négation, présupposé faux, injection de prompt, langue étrangère. Les
rubriques retrouvées sont affichées à chaque fois c'est ce qui permet de dire si l'échec
vient de la recherche ou de la génération.

Un contrôle de fidélité : Tout nombre présent dans la réponse doit figurer dans le contexte.
Heuristique simple, mais elle attrape la catégorie d'erreur la plus coûteuse pour un hôtel :
l'horaire ou le prix inventé avec aplomb.

## Les limites, et ce qui en est fait

| Limite | Origine | Traitée ? | Piste en production |
|---|---|---|---|
| Hallucination hors sujet | génération | oui seuil, refus sans LLM | recalibrer sur des questions réelles |
| Chiffre inventé | génération | détectée | refuser ou reformuler si alerte |
| Synonymes, fautes, langue | recherche | mesurée, hybride testé | embedding plus grand, reformulation |
| Question multi-rubriques | les deux | mesurée | `top_k` adaptatif, LLM plus grand |
| Question ambiguë | génération | mesurée | demander une précision au client |
| Injection de prompt | architecture | non limite structurelle | séparer système / utilisateur, aucune action sensible sans validation |
| Documentation obsolète | données | non | date par rubrique, réindexation |
| Évaluation sur peu de questions | méthode | assumée | jeu construit sur les vrais emails clients |

Honnêteté sur la mesure : 24 questions, c'est peu. Une question de plus ou de moins déplace
le hit@1 de plusieurs points. Ces chiffres donnent un ordre de grandeur, pas une certitude.

## L'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Interface de discussion avec historique, sources affichées avec leur score, alerte visible
quand une réponse contient un chiffre non sourcé, et réglage en direct du `top_k`, du seuil et
de la pondération α.

Le principe : le notebook sert à décider, le module sert à exécuter. Les valeurs calibrées
en partie 4 deviennent les réglages par défaut de `rag.py`.

Avant de déployer en ligne : les deux modèles pèsent environ 1,5 Go et tournent sur CPU.
Les hébergements gratuits ont des ressources limitées, ce qui peut rendre l'application lente
voire l'empêcher de démarrer. Pour une démonstration, l'exécution locale est la plus fiable ;
pour un usage réel, on remplacerait le petit LLM local par un appel à une API application plus
légère, réponses nettement meilleures.

## Installation

```bash
git clone https://github.com/badaouihakimou/hotel-rag-assistant.git
cd hotel-rag-assistant
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Les modèles se téléchargent automatiquement depuis Hugging Face au premier lancement
(environ 1,5 Go, mis en cache ensuite).

## Structure

```txt
├── notebook.ipynb      # les 5 parties
├── rag.py              # moteur RAG réutilisable (classe HotelRAG)
├── app.py              # interface Streamlit
├── utils.py            # affichage fourni par l'exercice
├── data/               # les 5 PDF de documentation
├── requirements.txt
└── README.md
```

## Modèles utilisés

| Rôle | Modèle | Pourquoi |
|---|---|---|
| Génération | `Qwen/Qwen2.5-0.5B-Instruct` | 500 M de paramètres, tourne sur un portable |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | multilingue la documentation est en français |

Le LLM est volontairement minuscule : les modèles de production en comptent des dizaines de
milliards. Ses maladresses sont donc attendues, et la partie 4 permet justement de distinguer
ce qui relève de sa taille de ce qui relève du système.

## Ce que le projet m'a appris

Un RAG se diagnostique en deux temps : Mesurer la recherche isolément, avant de juger la
génération.

Moins de contexte vaut mieux que plus : La partie 2 le démontre : noyé dans 1 400 mots, le
modèle rate des réponses pourtant présentes sous ses yeux.

Le refus est une fonctionnalité : Le seuil hors sujet garantit un « je ne sais pas » sans
dépendre de la bonne volonté du modèle cela se construit, cela ne s'espère pas.

Exercice issu du [Cahier de Vacances Data](https://machinelearnia.com/) de Machine Learnia
(Guillaume Saint-Cirgue). Les parties 4 et 5, ainsi que `rag.py` et `app.py`, sont les miennes.
