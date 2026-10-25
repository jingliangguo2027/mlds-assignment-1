"""Show the computed policy as a green/red grid, including a final-round zoom."""

from pathlib import Path
import runpy

from PIL import Image, ImageColor, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
result = runpy.run_path(str(ROOT / "main.py"))
V, T, MAX_PEOPLE = result["V"], result["T"], result["MAX_PEOPLE"]

GREEN = "#39AF78"
RED = "#E86C68"
INK = "#233044"
MUTED = "#596579"
FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def font(size):
    if Path(FONT).exists():
        return ImageFont.truetype(FONT, size)
    return ImageFont.load_default(size=size)


canvas = Image.new("RGB", (1500, 1370), "white")
draw = ImageDraw.Draw(canvas)
draw.text((150, 30), "班车调度：最优动作网格", font=font(38), fill=INK)
draw.rectangle((150, 102, 176, 128), fill=GREEN)
draw.text((188, 97), "0 = 不发车", font=font(25), fill=INK)
draw.rectangle((440, 102, 466, 128), fill=RED)
draw.text((478, 97), "1 = 发车", font=font(25), fill=INK)


def panel(first_t, last_t, top, height, title, detailed=False):
    left, width = 150, 1206
    columns = MAX_PEOPLE + 1
    rows = last_t - first_t + 1
    draw.text((left, top - 52), title, font=font(27), fill=INK)
    grid = Image.new("RGB", (columns, rows))
    colors = [ImageColor.getrgb(GREEN), ImageColor.getrgb(RED)]
    grid.putdata([
        colors[V[t][s][0]]
        for t in range(first_t, last_t + 1)
        for s in range(columns)
    ])
    canvas.paste(grid.resize((width, height), Image.Resampling.NEAREST), (left, top))

    # The enlarged panel shows every decision cell; the full panel uses major lines.
    step_s = 1 if detailed else 10
    for s in range(0, columns + 1, step_s):
        x = round(left + s * width / columns)
        draw.line((x, top, x, top + height), fill="#FFFFFF" if detailed else "#C0CCD0", width=1)
    boundaries = range(rows + 1) if detailed else range(0, rows + 1, 50)
    for row in boundaries:
        y = round(top + row * height / rows)
        draw.line((left, y, left + width, y), fill="#FFFFFF" if detailed else "#C0CCD0", width=1)
    draw.rectangle((left, top, left + width, top + height), outline=MUTED, width=1)

    ticks_t = list(range(first_t, last_t + 1)) if detailed else sorted(set([first_t, last_t] + list(range(100, last_t, 100))))
    for t in ticks_t:
        y = top + (t - first_t + 0.5) * height / rows
        draw.text((left - 18, y), str(t), font=font(21), fill=MUTED, anchor="rm")
    for s in range(0, columns, 25):
        x = left + (s + 0.5) * width / columns
        draw.text((x, top + height + 12), str(s), font=font(22), fill=MUTED, anchor="mt")
    draw.text((left + width / 2, top + height + 50), "等待人数 s", font=font(25), fill=INK, anchor="mt")
    draw.text((left - 18, top - 30), "时间 t", font=font(22), fill=MUTED, anchor="rm")


# Time T is a terminal boundary with no action, so display only decision rounds.
panel(0, T - 1, 210, 600, "全部 500 轮（从上往下：时间推进）")
panel(max(0, T - 10), T - 1, 1020, 240, "最后 10 轮放大", detailed=True)

output = ROOT / "policy_grid.png"
canvas.save(output)
print(output)
