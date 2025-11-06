pip install diffusers
pip install transformers
pip install accelerate


# Simple
python ai_fusion.py --a Charizard.png --b Psyduck.png --out Charizard_Psyduck.png


# Customize
python ai_fusion.py --a Charizard.png --b Psyduck.png --out Charizard_Psyduck.png \
    --prompt "anime-style electric ghost Pokémon with glowing purple aura" \
    --strength 0.7 --guidance 8


# following tests were done ...

python ai_fusion.py --a Charizard.png --b Mew.png --out Charizard_Mew.png
python ai_fusion.py --a Charizard.png --b Psyduck.png --out Charizard_Psyduck.png
python ai_fusion.py --a Charizard.png --b Mew.png --out Charizard_Mew_Electric.png --prompt "anime-style electric ghost Pokémon with glowing purple aura" --strength 0.7 --guidance 8