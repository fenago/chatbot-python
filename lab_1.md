
Text & Image Chatbot with Zero Data Using Humingbird 
====================================================





Let's power a chatbot backend that's smart visually and conversationally 
------------------------------------------------------------------------



![](./images/1_Arm6NbxTfBK7YJnxtntL3Q.jpeg)





Integrating AI into your project's app backend has become increasingly
popular in recent years. Many industries have adopted AI to improve
productivity, reduce costs, and streamline their app's processes. One of
these ways AI can improve an app's experience is by using a Chatbot. A
Chatbot is an automated agent that can interact with a user by talking
to them, responding to their requests, and generally being helpful. In
other words, Chatbots can make your user's experience much more natural
and seamless.

The unfortunate part is that chatbots can take a long time to build,
design and deploy. Gathering training data, building a model, or even
taking time to learn a service can be time-consuming. Fortunately,
Humingbird is here to save the day. Using Humingbird's `Text` method for text classification, we can apply the same
techniques to build a chatbot! For those that aren't aware of
Humingbird, I'd recommend you check out the original blog post here; The
TLDR is that you can easily build ML classifiers with no data or
training.

With that being said, I think it's time to build a chatbot!





Building a chatbot with image recognition capabilities 
======================================================

![](./images/1_eSPqeUsA4V5lYSypAI6B7g.png)

Our project outline is simple: we're going to build a chatbot for a
fictional storefront that sells ice cream. This chatbot will be able to:

- Respond to general user queries about ice cream
- Recognize images of different ice cream flavors and respond
    accordingly
- Be able to fallback if the chatbot doesn't have a high enough
    confidence score

*Side note*: While this is simple in nature and somewhat of a fun
example, this project outline could be used in many applications. Fusing
together visual and conversational abilities into a single platform
could help automate a number of different tasks, like automated customer
service.

To continue with the rest of this lab, let\'s install the
Humingbird package with the command:

```
pip install humingbird
```

Step 1: Building our intent recognition system 
----------------------------------------------

First, we need to start by building an intent recognition system. For
those not familiar, **intent recognition is the task of predicting what
a query "means".** In other words, we set out a pre-defined map of what
might be said to our chatbot given the application.

For our ice cream chatbot, we're going to use the following
`intents`:

```
# intents[greeting, goodbye, menu, prices, start_order]
```

All of these are fairly self-explanatory on what their "roles" will be.

To build our intent recognizer, let\'s use the following snippet using
Humingbird (filename: `intent_recognition.py`):

```
import humingbird

intent_recognition = humingbird.Text.predict(
  text="I was wondering what you have on your menu? I love ice cream!",
  labels=["greeting", "goodbye", "menu", "prices", "start_order"]
)

print(intent_recognition)
```



This will give us the following output:

```
[
  {
    "className": "menu",
    "score": 0.84,
  },  {
    "className": "greeting",
    "score": 0.09
  },
  
  {
    "className": "prices",
    "score": 0.06,
  },  {
    "className": "goodbye",
    "score": 0.01
  },  {
    "className": "start_order",
    "score": 0.01
  }
]
```

Awesome! Our intent recognition system correctly predicted that the
output was the user wanting to see the menu. There's one issue: we don't
have any responses! Let's build that.

Step 2: Build a response system 
-------------------------------

Don't sweat, we can use the following Python dictionary for our response
library  (filename: `responses.json`):

```
{
  "greetings": ["Hi! Welcome to the ice cream shop.", "Welcome! What can i help you with?"],
  "goodbyes": ["Goodbye!", "Nice chatting! Have a great day."],
  "menu": ["Here is our menu: https://www.randomicecream.co"],
  "prices": ["Our prices our affordable for all!", "We have super low prices! Pracitcally free!"],
  "start_order": ["Lets start the order here: https://www.fakepaymentlink.com"]
}
```



Now we can build a simple function to recognize intents and respond
accordingly (filename: `response_function.py`):

```
import humingbird
import random

intents = {
  "greeting": ["Hi! Welcome to the ice cream shop.", "Welcome! What can i help you with?"],
  "goodbye": ["Goodbye!", "Nice chatting! Have a great day."],
  "menu": ["Here is our menu: https://www.randomicecream.co"],
  "prices": ["Our prices our affordable for all!", "We have super low prices! Pracitcally free!"],
  "start_order": ["Lets start the order here: https://www.fakepaymentlink.com"]
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


detect_and_respond("Hi there! How are you today?")
```




Which will generate a response from our chatbot, such as:

```
Welcome! What can i help you with?
```

At this point, we have everything we need for a text-based chatbot. We
can improve our chatbot by adding more `intents`
and better responses.

But we want to add one more piece: visual capabilities.

Step 3: Adding visual capabilities to our chatbot 
-------------------------------------------------

Up until this step, we have built a simple text-based chatbot. While we
have saved lots of time and abstracted away a lot of code, we haven't
done anything different than what most chatbot platforms can do.

We are going to take a step in a different direction by adding a visual
understanding component to our chatbot. In our simple example of an ice
cream store chatbot, we're going to recognize different flavors of ice
cream and respond with a fictional checkout link.

To do this with Humingbird, we can use the following code snippet
(filename: `ice_cream.py`):


```
import humingbird

prediction = humingbird.Image.predict(
    image_path='ice-cream.jpg',
    labels=["strawberry ice cream", "vanilla ice cream", "chocolate ice cream"]  # add more if you'd like :)
)

print(prediction)
```



We can use this image:

![](./images/1_Eeq7zU8PFIYOKGwI4zsSGA.jpeg)

And our code snippet will return:

```
[  {
    "className": "strawberry ice cream"
    "score: 0.93
  },  {
     "className": "vanilla ice cream",
     "score": 0.05,
  },  {
     "className": "chocolate ice cream"
     "score": 0.02
  }]
```

Awesome! We've done the toughest part with only a few lines of code.
Now, let\'s put it all together by adding some "visual intents" with a
`visual_intent_detection` function (filename: `updated_responses.py`):

```
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
```


Which will return (with the example image above):

```
Great choice! Here is the checkout link: https://www.fakeicecream.com/checkout/strawberry
```



Conclusion 
==========

In this lab, we build a very basic text + image-based chatbot using
Humingbird. We saved tons of time not collecting data, abstracted away
lots of complicated code and we needed zero training to build this
multimodal model. While this chatbot could be improved and the concept
is a bit silly, chatbots that respond textually and visually have a huge
application. Imagine if we could build a visual chatbot for students to
understand medical images, or to detect landmarks and learn about their
history?
