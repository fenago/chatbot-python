import humingbird

prediction = humingbird.Image.predict(
    image_path='ice-cream.jpg',
    labels=["strawberry ice cream", "vanilla ice cream", "chocolate ice cream"]  # add more if you'd like :)
)

print(prediction)