# TODO

1. clear all " # TODO:" s, " # FIXME:" s from the code
2. Add more augmentations
3. Finalize the frontend demo - what options do we want to add
4. Merge in Virtual Scribe
5. Decide on sensible defaults for the word augmentations - fix up rotation issue on long words
6. Come up with name for open sourcing and replace typesetter everywhere with the name settled on
7. Create Providers for Fonts, Background images etc - faker providers maybe? - Move all data (backgrounds, fonts) into S3 and store in ~/.typesetter
8. Allow the production of images to be reproducible using random seeds - and unit test this is working
9. ReadMe Update + Include high level software architecture diagram in docs (how to blend image augs, fonts, backgrounds, mnist, dictionaries, nlpaug, handwritten etc)
10. Add documentation into GitHub on CapgeminiInventIDE
11. Move data generation etc out into separate repo to be used very rarely
12. Allow fonts to be generated with multiple colors
13. Allow fonts to be generated using textured colors - use the font as a mask to extract from a textured background
14. Handle spaces and new lines to produce full images
