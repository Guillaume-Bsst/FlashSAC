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
- IsaacLab renvoie les **mêmes tenseurs** `terminated` / `time_out` à chaque pas : les copier avant de les garder.

## E0

### Hypothèse (écrite avant)

### Constat (écrit après)

## E1

### Hypothèse (écrite avant)

### Constat (écrit après)
