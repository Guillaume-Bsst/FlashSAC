# Labo Ant : imitation off-policy rapide à partir d'une référence cinématique — design

## Pourquoi

L'imitation d'un mouvement retargeté (OT retarget, dépôt holonew) à la BeyondMimic (PPO) prend plus de 5 h,
même en locomotion simple. L'objectif final est de passer **sous 1 h, idéalement 30 min, sur G1** (une RTX 5070 Ti).
L'idée : remplacer PPO par FlashSAC (off-policy, déjà plus rapide que PPO en pratique) et **exploiter le replay
buffer** avec la référence. La littérature off-policy de tracking (FastSAC/Holosoma, QF3 ~30 min sur G1) n'utilise
la référence que pour les départs (RSI), jamais dans le buffer : c'est l'angle recherche.

Avant G1, on construit un **labo sur Ant** : petit, rapide, où l'on peut isoler chaque idée et garder la vérité
terrain. Le labo sert aussi à former l'intuition de l'utilisateur : chaque expérience doit rendre lisible
*pourquoi* un résultat apparaît.

## Objectifs

1. Une tâche de **tracking d'une référence cinématique** sur Ant (IsaacLab), utilisable à l'identique par PPO et FlashSAC.
2. Mesurer le **facteur d'accélération** (temps réel jusqu'au succès) de FlashSAC sur PPO.
3. Tester, un levier à la fois :
   - levier 1 : FlashSAC au lieu de PPO ;
   - levier 2 : action résiduelle autour de la référence (`cible PD = q_ref(t) + δ`, à la DReCon) ;
   - levier 3 : la référence **dans le buffer** (voir plus bas), contre le RSI seul.
4. Tracer le gain du levier 3 en fonction du **degré d'infaisabilité** de la référence.

## Non-objectifs

- Pas de G1 dans ce projet (il viendra après, avec ce qu'on aura appris).
- Pas d'objets, pas de tâche + style (commande de vitesse avec style conservé) : extensions possibles plus tard.
- Pas d'OTRetarget sur Ant (il est construit pour humain → humanoïde, SMPL-X).
- Pas de sim2real.

## Anti-objectifs (ce qui ferait échouer le projet même si « ça marche »)

- **Une comparaison déloyale** : PPO mal réglé, ou FlashSAC réglé sur les résultats. Chaque algorithme garde sa
  recette standard, fixée **avant** de voir les résultats.
- **Une référence dynamiquement parfaite** : trop facile, elle ne dirait rien du cas retargeté.
- **Des conclusions qui ne transfèrent pas au G1** : par exemple un Ant commandé en couples alors que le G1 l'est en PD.
- **Changer plusieurs choses à la fois** entre deux expériences.
- **Des résultats illisibles** pour l'utilisateur : chaque expérience doit avoir son hypothèse écrite avant, et son
  constat après.

## Décisions de l'utilisateur

- Ant, dans **IsaacLab** (pile `hssim` : IsaacSim 5.1 / IsaacLab 2.3.0, `~/.holosoma_deps`), comme les runs G1 existants.
- Référence de **type A** : un expert Ant dont on enregistre la cinématique, puis qu'on dégrade.
- Les défauts à imiter, d'après les retargetings OT sur G1 : la référence suit l'humain, qui a **plus de capacités
  que le robot** (force, couple, vitesse) ; les **pas sont parfois trop grands** (pas de scaling). En locomotion,
  le retargeting ne pose « quasi jamais » de problème : la référence réelle est presque faisable.
- **Commande en PD** acceptée (« PD ça me va »).
- L'utilisateur veut **comprendre chaque étape** : il doit être interrogé et associé aux choix, pas seulement recevoir
  du code.

## Décisions prises par Claude (à vérifier par l'utilisateur)

- **PD plutôt que couples** : même espace d'action que le G1 (transfert des conclusions), nécessaire au levier 2,
  et buffer initial plus sain (une action aléatoire reste une posture tenue). Gains choisis selon la règle de
  BeyondMimic (fréquence naturelle / amortissement à partir de l'armature), adaptée à Ant.
- **La « force » d'un Ant** : un facteur `S` qui multiplie à la fois `Kp`, `Kd` et la limite de couple
  (des moteurs plus gros, même loi de commande). `S = 1` = l'Ant d'IsaacLab (7,5 N·m max par articulation).
- **L'expert** : **un seul**, de force `S_e = 1,5`, entraîné avec FlashSAC sur la reward « avancer »
  (celle d'`Isaac-Ant-Direct-v0`), en PD. On enregistre sa cinématique (racine + 8 articulations) et ses couples
  sur un clip de 10 s de marche en ligne droite.
- **Les boutons de dégradation** :
  - `k_cap` (écart de capacités, le bouton principal) : l'Ant qui suit a la force `S_e / k_cap`. On fait varier
    la force du suiveur plutôt que celle de l'expert : la référence reste **la même** d'un réglage à l'autre, donc
    seule l'infaisabilité change. `k_cap = 1` = référence faisable (cas témoin).
  - `σ` (bruit lent du tube, même forme que `slow_noise` de `holonew/OTRL/otrl/toys/tube.py`, ℓ = 0,5 s) sur les
    positions articulaires et la pose de la racine, pour la diversité et pour l'analyse d'infaisabilité.
  - Valeurs : `k_cap ∈ {1,0 ; 1,25 ; 1,5}`. Réglage par défaut : `k_cap = 1,25`, et `σ` au niveau « plancher »
    de holonew (D26, r2p1t1 : 2° sur les articulations et la rotation de la racine, 1 cm sur sa position),
    pour coller au cas réel (presque faisable).
- **Mesure de l'infaisabilité d'une référence**, trois façons, à rapporter pour chaque réglage :
  - **rejeu en boucle ouverte** : PD qui suit `q_ref` sans politique, depuis le RSI ; durée moyenne avant terminaison
    (mesure tout à la fois, mais n'est pas parfaite même pour une référence faisable : sans politique, les erreurs
    s'accumulent) ;
  - **résidu de contact** : pour un robot à pattes, le sol fournit l'essentiel de la force sur le corps. Ce qui est
    infaisable, c'est la part de la force et du couple nécessaires (dynamique du centre de masse) que **les pieds
    posés ne peuvent pas fournir** : le sol pousse mais ne tire pas, seulement par les pieds au sol, dans le cône
    de frottement (μ = 1). Calcul par moindres carrés positifs (NNLS) sur les forces de contact. C'est ce que le
    bruit `σ` dégrade (pieds qui glissent ou flottent) ;
  - **demande de couple** : part des frames où les couples enregistrés de l'expert dépassent la limite du suiveur.
    C'est ce que `k_cap` dégrade.
- **Recette de tracking = BeyondMimic adaptée à Ant**, pour que la baseline PPO soit celle qu'on veut battre :
  - ancre = torse ; corps suivis = torse + segments des pattes (pieds compris) ;
  - observation : proprioception (articulations, vitesse angulaire de base, gravité projetée, dernière action) +
    référence de la frame suivante (articulations, vitesses) + erreur de pose de l'ancre en repère du torse ;
  - reward : somme de termes `exp(-erreur²/σ²)` (position et orientation de l'ancre, positions et orientations des
    corps, vitesses linéaires et angulaires) + pénalité de variation d'action ;
  - terminaison : écart de hauteur ou d'orientation de l'ancre, ou de hauteur des pieds, au-delà d'un seuil ; fin du clip ;
  - départs : RSI avec **échantillonnage adaptatif** des frames selon les échecs (BeyondMimic) ;
  - action (baseline) : `cible PD = posture par défaut + échelle · a`.
- **PPO** : rsl_rl 3.0.1 avec la recette BeyondMimic déjà écrite dans `holonew/OTRL/otrl/ppo_dict.py`.
- **FlashSAC** : la recette des runs G1 (`run_g1.sh` : batch 2048, n_step 3, UTD 2, AMP), sans réglage spécifique.
- **Critère de succès d'un run** : ≥ 90 % des épisodes d'évaluation vont au bout du clip sans terminaison, et
  l'erreur moyenne de tracking des articulations passe sous 0,1 rad (seuil provisoire, à revoir après E0).
  On rapporte le **temps réel** et le **nombre de pas d'environnement** pour l'atteindre, sur ≥ 3 graines, même GPU,
  runs lancés l'un après l'autre. Plafond de 3 h par run : au-delà, le run compte comme un échec.
- **Levier 2 aussi sur PPO** (optionnel) si E2 montre un gros gain, pour savoir si c'est l'algorithme ou l'action
  qui fait la différence.

## Le levier 3 : la référence dans le buffer

Trois façons de transformer la référence en transitions, toutes comparées au **RSI seul** :

| Variante | États | Actions | Physique |
|---|---|---|---|
| RSI seul (témoin) | départs sur la référence | celles de la politique courante | vraie |
| Référence brute | toute la référence | dynamique inverse → cible PD | celle de la référence (peut être fausse) |
| Référence projetée | reset sur la frame, 1 pas de simulation | dynamique inverse → cible PD | vraie |
| Référence corrigée | toute la référence | dynamique inverse → cible PD | reward pénalisée par le résidu de contact (esprit DARC) |

- Les transitions de référence vivent dans un **buffer séparé** (pas écrasées par la FIFO), avec une **part dans
  chaque batch qui décroît** au fil de l'entraînement.
- Attention à FlashSAC : la BatchNorm est calculée sur le batch mélangé, et le scaling de reward suit les retours
  observés ; la part de référence modifie les deux.
- Conversion couple → cible PD : `q_cible = q + (τ + Kd·q̇) / Kp`, puis ramenée dans l'espace d'action normalisé.
- La dynamique inverse doit utiliser **le même modèle** que le simulateur (pas l'Ant de gymnasium) :
  `τ = M(q)·q̈ + h(q, q̇) − J_cᵀ·f` avec `q̈` par différences finies lissées sur la référence et `f` les forces de
  contact estimées par le calcul du résidu de contact (sans elles, les couples des jambes seraient faux : ce sont
  elles qui portent le robot).

## Expériences, dans l'ordre

Chaque expérience a une **hypothèse écrite avant** et un **constat écrit après**, dans un journal en français
(`docs/ant_lab/JOURNAL.md`), pour que l'utilisateur suive le raisonnement.

| # | Expérience | Ce qu'elle isole |
|---|---|---|
| E0 | Expert entraîné, référence enregistrée ; rejeu en boucle ouverte, résidu de contact et demande de couple pour chaque `k_cap` et `σ` | la référence est-elle « presque faisable » comme sur G1 ? |
| E1 | PPO vs FlashSAC, référence par défaut | levier 1 |
| E2 | FlashSAC + action résiduelle | levier 2 |
| E3 | FlashSAC + chaque variante du levier 3 | levier 3 contre RSI |
| E4 | Meilleure variante de E3, balayage de `k_cap` (et `σ`) | la courbe « utilité vs infaisabilité » |

On s'arrête et on en parle avec l'utilisateur après E0 et après E1, avant d'aller plus loin. Le premier plan
d'implémentation couvre tout jusqu'à E1 ; E2 à E4 auront leur propre plan, écrit après la discussion d'E1.

## Laissé au builder

Emplacement des fichiers (dans l'esprit des envs existants : `flash_rl/envs/…_isaaclab_env.py`, `register()`,
entrée dans `ACTION_BOUNDS`, scripts `run_*.sh`), valeurs exactes des gains, seuils, `σ` de reward, nombre
d'environnements par algorithme (chacun selon sa recette), journalisation.

## Risques

- **Ant trop facile** : si PPO réussit en quelques minutes, le facteur d'accélération dit peu. Parade : clip plus
  dur (allure plus rapide, virages), `k_cap` plus grand.
- **Gait de l'expert bizarre** (sauts fréquents chez Ant) : le vérifier visuellement à E0.
- **API de dynamique inverse dans IsaacLab** : matrice de masse et termes gravité/Coriolis à vérifier sur la
  version installée.
- **Mélange démo/agent avec la BatchNorm** de FlashSAC : peut fausser le levier 3 indépendamment de l'idée.
