import humingbird

intent_recognition = humingbird.Text.predict(
  text="I was wondering what you have on your menu? I love ice cream!",
  labels=["greeting", "goodbye", "menu", "prices", "start_order"]
)

print(intent_recognition)