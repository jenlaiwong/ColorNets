from subprocess import Popen, PIPE

fonts = ["AndaleMono",
"AppleChancery",
"AppleMyungjo",
"Arial",
"ArialB",
"ArialBI",
"ArialBk",
"ArialI",
"ArialNarrow",
"ArialNarrowB",
"ArialNarrowBI",
"ArialNarrowI",
"ArialRoundedB",
"ArialUnicode",
"Ayuthaya",
"BigCaslonM",
"BrushScriptI",
"Chalkduster",
"ComicSans",
"ComicSansMSB",
"CourierNew",
"CourierNewB",
"CourierNewBI",
"CourierNewI",
"GB18030Bitmap",
"Georgia",
"GeorgiaB",
"GeorgiaBI",
"GeorgiaI",
"GungSeo",
"Gurmukhi",
"HeadLineA",
"Herculanum",
"HoeflerTextOrnaments",
"Impact",
"InaiMathi",
"Kokonor",
"Krungthep",
"MicrosoftSansSerif",
"Osaka",
"OsakaMono",
"PCMyungjo",
"PilGi",
"PlantagenetCherokee",
"Sathu",
"Silom",
"Skia",
"Tahoma",
"TahomaB",
"TimesNewRoman",
"TimesNewRomanB",
"TimesNewRomanBI",
"TimesNewRomanI",
"Trebuchet",
"TrebuchetMSB",
"TrebuchetMSBI",
"TrebuchetMSI",
"Verdana",
"VerdanaB",
"VerdanaBI",
"VerdanaI",
"Zapfino"]


alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
rotations = [0, 90, 180, 270]

commands = []

for letter in alphabet:
    for font in fonts:
        for rotation in rotations:
            filename = "renders/{}_{}_{}.png".format(letter, font, rotation)
            commands.append(
                [
                    "convert",
                    "-rotate",
                    str(rotation),
                    "-size",
                    "10x10",
                    "-font",
                    font,
                    "-pointsize",
                    "10",
                    "label:{}".format(letter),
                    filename
                ])

def run_parallel(cmds):
    parallel_process = Popen(['parallel'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    joined = "\n".join([" ".join(cmd) for cmd in cmds])
    parallel_process.communicate(input=joined)[0]

run_parallel(commands)
