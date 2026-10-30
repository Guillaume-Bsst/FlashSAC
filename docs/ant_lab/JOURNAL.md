# Journal du labo Ant

## Comment lire ce journal

Chaque expérience a deux parties. L'**hypothèse** est écrite *avant* de lancer quoi que ce soit : ce qu'on
s'attend à voir et pourquoi. Le **constat** est écrit *après* : ce qu'on a vu, les chiffres, et ce qui confirme ou
contredit l'hypothèse. L'écart entre les deux, c'est là que l'intuition se construit. Le design est dans
`docs/superpowers/specs/2026-10-30-ant-tracking-lab-design.md`.

## Notes techniques

- **Modèle Ant (sonde, 2026-10-30)** : 9 corps (`torso`, 4 `*_leg`, 4 `*_foot`), 8 articulations (hanches `*_leg`
  dans ±0,70 rad, chevilles `*_foot` dans [0,52 ; 1,75] rad en valeur absolue, posture par défaut ±0,785).
  Masse totale **0,911 kg** (torse 0,484 kg). Au repos en PD, torse à 0,435 m et bouts de pieds au sol à 0,2 mm
  près : les offsets de `nv_ant.xml` sont justes.
- **API de dynamique PhysX disponible** : matrice de masse (14×14 = 6 de base flottante + 8 articulations),
  jacobiennes (9 corps × 6 × 14), gravité et Coriolis compensés. La dynamique inverse du levier 3 est donc faisable
  avec le même modèle que le simulateur.
- Ordre de grandeur : à force 1, un couple de 7,5 N·m sur un robot de 0,9 kg, c'est beaucoup (le poids vaut ~9 N).
  À vérifier à E0 : `k_cap` crée-t-il vraiment une demande de couple au-delà de la limite du suiveur ?
- **Débit FlashSAC (essai, env expert, 256 envs, 2 mises à jour par pas)** : ~57 it/s une fois l'entraînement lancé,
  soit ~15 000 pas d'env/s. Avec 2048 envs, le débit sera limité par les mises à jour du réseau (batch 2048 fixe).
- `flash_rl/types.py` du commit importe `jax`, absent de l'env `hssim` : la branche reprend le correctif de
  l'utilisateur (`jax` optionnel).
- **Env de tracking vérifié** : robot posé sur la référence → les 6 termes de suivi valent 1,0 (env 0 et dernier env),
  fin de clip exacte, 300 pas sans NaN.
- **À discuter à E0 — pénalité de limite articulaire** : la recette BeyondMimic pénalise (×10) toute articulation
  au-delà de 90 % de sa plage. Or l'expert Ant colle ses chevilles en butée (cible PD jusqu'à 0,285 rad pour une
  butée à 0,52) : sur la référence d'essai, suivre *parfaitement* coûte déjà 1,6 point de reward sur 5. Une reward
  qui punit le suivi parfait, c'est un signal contradictoire. À mesurer sur la vraie référence (part des frames
  hors zone souple) avant de décider.
- **PPO d'essai** (tracking, 512 envs, 60 itérations, GPU partagé avec l'entraînement de l'expert) : 0,66 s par
  itération de 24 pas, soit ~18 600 pas d'env/s. Les vrais débits seront mesurés sur GPU libre à E1.
- **Bug trouvé : le PD « implicite » de PhysX ne fait rien sur cet Ant.** Avec `ImplicitActuatorCfg`, PhysX a bien
  les raideurs, IsaacLab affiche un couple saturé à 11,25 N·m, mais l'articulation ne bouge pas d'un millième de
  radian : actions nulles ou à fond donnent des trajectoires identiques au bit près. On l'a vu parce que tous les
  checkpoints de PPO et de FlashSAC avaient exactement le même score. Correction : PD **explicite**
  (`IdealPDActuatorCfg`, mêmes Kp, Kd et limite, couple calculé à 120 Hz et envoyé en effort, le chemin qu'utilise
  l'Ant d'origine). Le premier expert, entraîné jambes figées, est jeté. Au repos en PD, le torse est maintenant à
  0,48 m.
- **Expert (PD explicite, force 1,5)** : FlashSAC, 2048 envs, 30 M de pas d'env (14 648 itérations), **6,1 min**
  de bout en bout (dont le démarrage d'IsaacSim et le remplissage du buffer). Clip de 10 s enregistré au checkpoint
  final : **4,16 m/s** en ligne droite, dérive de lacet 0,03 rad, aucune chute, couple max 11,25 N·m (= sa limite :
  l'expert sature ses moteurs). Références : `ref_s0` (propre) et `ref_s1` (bruit plancher), 601 frames chacune.
- IsaacLab renvoie les **mêmes tenseurs** `terminated` / `time_out` à chaque pas : les copier avant de les garder.

## E0

### Hypothèse (écrite avant)

Question : la référence par défaut (k = 1,25, bruit « plancher ») est-elle *presque faisable*, comme un
retargeting OT en locomotion ? Trois mesures, chacune sensible à un défaut différent :

- **Demande de couple** (ne dépend que de k) : à k = 1, le suiveur a la force de l'expert, donc 0 % de frames
  au-delà de sa limite. Quand k monte, la limite du suiveur baisse (11,25 → 9 → 7,5 N·m) : la part de frames où
  l'expert a demandé plus que ça monte. Comme l'expert sature souvent ses moteurs (c'est ce que fait une politique
  RL qui pousse fort), je m'attends à une part **non négligeable dès k = 1,25** (de l'ordre de 10 à 30 %).
- **Résidu de contact** (ne dépend que du bruit σ) : sur la référence propre (σ = 0), faible mais pas nul (contacts
  mous du simulateur, différences finies, seuil de contact de 2 cm) : médiane de quelques % de M·g. Le bruit lent
  bouge les pieds par rapport au sol (pieds qui flottent ou s'enfoncent), donc le résidu monte avec σ, surtout au
  p95 (les instants où aucun pied ne « touche » alors que le corps accélère).
- **Rejeu en boucle ouverte** (tout à la fois) : même sur la référence propre à k = 1, la survie est limitée, car
  sans politique les petites erreurs s'accumulent (l'Ant n'est pas stable passivement en marchant). Elle baisse
  quand k et σ montent.

Ce qui me ferait changer d'avis : une demande de couple nulle pour tous les k (Ant surpuissant, il faudrait des k
plus grands), ou une survie en boucle ouverte déjà nulle à k = 1, σ = 0 (le rejeu ne discrimine rien).

### Constat (écrit après)

## E1

### Hypothèse (écrite avant)

### Constat (écrit après)
