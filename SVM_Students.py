
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('persons_pics_train.csv')
df.head()
def get_img_by_row(row):
  return row.drop('label').astype(float).to_numpy().reshape(62,47), row['label']
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    img, lbl = get_img_by_row(df.iloc[i])
    axi.imshow(img, cmap='gray')
    axi.set(xticks=[], yticks=[],
            xlabel=lbl.split()[-1])
plt.savefig('persons_pics_img_for_description.png', dpi = 300, bbox_inches='tight')
plt.show()