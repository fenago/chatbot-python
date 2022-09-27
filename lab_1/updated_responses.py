import humingbird
import random

intents = {
  "greeting": ["Hi! Welcome to the ice cream shop.", "Welcome! What can i help you with?"],
  "goodbye": ["Goodbye!", "Nice chatting! Have a great day."],
  "menu": ["Here is our menu: https://www.randomicecream.co"],
  "prices": ["Our prices our affordable for all!", "We have super low prices! Pracitcally free!"],
  "start_order": ["Lets start the order here: https://www.fakepaymentlink.com"]
}

visual_intents = {
  "strawberry ice cream": ["Great choice! Here is the checkout link: https://www.fakeicecream.com/checkout/strawberry"],
  "vanilla ice cream": ["Great choice! Here is the checkout link: https://www.fakeicecream.com/checkout/vanilla"],
  "chocolate ice cream": ["Great choice! Here is the checkout link: https://www.fakeicecream.com/checkout/chocolate"],
}


def detect_and_respond(query):
  """Detects an intent from the users query and returns a response from the most likely intent"""
  
  prediction = humingbird.Text.predict(
    text=query,
    labels=["greeting", "goodbye", "menu", "prices", "start_order"]
  )
  
  highest_score = 0
  highest_score_class = ""
  
  for i in prediction:
      if i["score"] > highest_score:
          highest_score = i["score"]
          highest_score_class = i["className"]
  
  return random.choice(intents[highest_score_class])


def visual_intent_detection(image):
  """Detects an intent from the users image and returns a checkout response"""
  
  prediction = humingbird.Image.predict(
    image_path=image,
    labels=["strawberry ice cream", "vanilla ice cream", "chocolate ice cream"]
  )
  
  highest_score = 0
  highest_score_class = ""
  
  for i in prediction:
      if i["score"] > highest_score:
          highest_score = i["score"]
          highest_score_class = i["className"]
  
  return random.choice(visual_intents[highest_score_class])