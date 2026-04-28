# ============================================================
# File: src/create_dataset.py
# Purpose: Generate a realistic synthetic social media dataset
# ============================================================

import pandas as pd
import random
import os

# ── Seed for reproducibility ──────────────────────────────
random.seed(42)

# ── Raw text pools per brand / sentiment ─────────────────
POSITIVE_POSTS = [
    "Absolutely love this product! Works perfectly every time.",
    "Amazing customer service, resolved my issue in minutes!",
    "Best purchase I've made this year. Highly recommended!",
    "The quality is outstanding. Will definitely buy again.",
    "Exceeded my expectations. Five stars all the way!",
    "Super fast delivery and the packaging was pristine.",
    "Their app update is so smooth and intuitive now.",
    "Fantastic experience from start to finish. Bravo!",
    "The new feature is exactly what I needed. Great job team!",
    "Never been happier with a brand. Top-notch service.",
    "Wow, just wow! This exceeded all my expectations.",
    "The support team went above and beyond to help me.",
    "Loving the new interface. Clean, fast, and beautiful.",
    "Product arrived early and works flawlessly. Impressed!",
    "This is the best app on my phone right now, hands down.",
    "Incredible value for money. Cannot believe how good this is.",
    "The team actually listens to feedback. Rare and appreciated.",
    "My order was perfect. Great quality and quick shipping.",
    "Really happy with the results. This brand truly delivers.",
    "Outstanding product and even better after-sales support.",
    "Zomato delivered my food in 20 minutes! Still hot. Amazing.",
    "Netflix new season is absolutely binge-worthy. 10/10!",
    "Swiggy pro membership is totally worth it. Love the discounts.",
    "Amazon Prime delivery is unbeatable. Got it same day!",
    "Flipkart Big Billion Days was insane. Got a great deal!",
    "My bank's new app is so easy to use. No more branch visits.",
    "The new phone camera is phenomenal. Photos look professional.",
    "Great experience at the service center. Fast and honest.",
    "This skincare product cleared my skin in two weeks!",
    "The earphones have crystal clear sound and deep bass. Love it.",
]

NEGATIVE_POSTS = [
    "Terrible experience. Product stopped working after two days.",
    "Customer support is absolutely useless. No help at all.",
    "Ordered a week ago and still no delivery. This is unacceptable.",
    "Complete waste of money. Quality is pathetically poor.",
    "Worst purchase ever. Totally disappointed and frustrated.",
    "The app keeps crashing every time I try to check out.",
    "Got a damaged product and refund request was rejected. Disgusting.",
    "Their service is a joke. Nobody responds to my complaints.",
    "Charged twice for one order and no resolution in sight.",
    "Product looks nothing like the photos. Total scam.",
    "Three weeks and my issue is still unresolved. Shameful.",
    "The food arrived cold and late. Never ordering from here again.",
    "Hidden charges ruined my trust in this brand completely.",
    "Support bot just loops the same useless answers. So frustrating.",
    "Returned the item but still waiting for my refund after 15 days.",
    "This brand used to be good. Now the quality has plummeted.",
    "Absolute disaster of an app update. Broke everything I used.",
    "Their so-called 'premium' service is worse than the free one.",
    "Delivery was marked delivered but nothing arrived. Big scam.",
    "Filed three complaints. Not a single response. Pathetic.",
    "Zomato took 90 minutes and the food was completely wrong.",
    "Netflix removed my favourite show without any notice. So angry.",
    "Swiggy delivery guy was rude and the food was spilled.",
    "Amazon sent me a counterfeit product. Absolutely unacceptable.",
    "Flipkart cancelled my order on the delivery day. Infuriating.",
    "Bank app locked me out and customer care kept me on hold for 2 hours.",
    "The earphones broke within a week. Cheap quality for a high price.",
    "Service center took my phone for a week and fixed nothing.",
    "This skincare cream caused a rash. Terrible product.",
    "The new update removed features I relied on daily. Horrible.",
]

NEUTRAL_POSTS = [
    "Just received my order. Will update after I use it for a while.",
    "The product is okay. Nothing special but does the job.",
    "Delivery was on time. Packaging was standard.",
    "Customer service responded. Waiting for the actual fix.",
    "Tried the new feature. It's different, still getting used to it.",
    "The app was updated today. Looks a bit changed.",
    "Placed an order for the new model. Will review once it arrives.",
    "Product quality seems decent so far. Too early to say.",
    "Watched the new show. It's average. Not bad, not great.",
    "The pricing is on par with competitors. Nothing stands out.",
    "Received a replacement. Testing it now to check if it works.",
    "First time using this brand. Will share thoughts after a week.",
    "The service center booked my appointment for next Thursday.",
    "Updated the app as prompted. No crashes so far.",
    "The subscription renewed automatically. Price unchanged.",
    "Got the notification about the new collection. Will check later.",
    "Ordered two items, one is out of stock. Waiting for an update.",
    "The interface is slightly different. Takes some getting used to.",
    "Spoke to support, they escalated the ticket. Now waiting.",
    "The product came with a user manual. Looks fairly straightforward.",
    "Tried the new Zomato feature. Seems useful. Let's see.",
    "Netflix added some new content. Nothing that interests me yet.",
    "Swiggy app updated with new design. Will try ordering later.",
    "Amazon is having a sale this weekend. Checking out the deals.",
    "Flipkart sent me a voucher. Not sure what to spend it on.",
    "Bank sent a notification about a new service. Haven't read it.",
    "The phone has good specs on paper. Will know more after using it.",
    "Service was neither fast nor slow. Just normal.",
    "The new season is out. Will watch over the weekend.",
    "Placed a return request. Waiting to see how the process goes.",
]

BRANDS = [
    "Amazon", "Flipkart", "Zomato", "Swiggy", "Netflix",
    "HDFC Bank", "Paytm", "Samsung", "OnePlus", "Myntra",
    "Ola", "Uber", "MakeMyTrip", "boAt", "Nykaa"
]

PLATFORMS = ["Twitter", "Instagram", "Facebook", "Reddit", "App Store", "Google Play"]

def generate_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Create a balanced synthetic social media dataset."""

    records = []
    n_each = n_samples // 3          # equal split across sentiments
    remainder = n_samples - n_each * 3

    categories = (
        [("positive", POSITIVE_POSTS)] * n_each +
        [("negative", NEGATIVE_POSTS)] * n_each +
        [("neutral",  NEUTRAL_POSTS )] * (n_each + remainder)
    )

    for sentiment, pool in categories:
        base_text = random.choice(pool)

        # Small variation: prefix brand or add hashtag occasionally
        brand = random.choice(BRANDS)
        platform = random.choice(PLATFORMS)
        likes = random.randint(0, 5000) if sentiment == "positive" else random.randint(0, 1000)
        retweets = random.randint(0, likes // 2 + 1)

        # Randomly prepend @brand mention
        text = base_text
        if random.random() > 0.5:
            text = f"@{brand.replace(' ','')} " + text
        if random.random() > 0.7:
            text += f" #{brand.replace(' ','')}"

        records.append({
            "id"       : len(records) + 1,
            "platform" : platform,
            "brand"    : brand,
            "text"     : text,
            "sentiment": sentiment,
            "likes"    : likes,
            "retweets" : retweets,
        })

    random.shuffle(records)
    df = pd.DataFrame(records)
    df["id"] = range(1, len(df) + 1)   # re-assign after shuffle
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(1000)
    path = os.path.join("data", "social_media_posts.csv")
    df.to_csv(path, index=False)
    print(f"✅ Dataset saved → {path}")
    print(f"   Total rows : {len(df)}")
    print(f"   Columns    : {list(df.columns)}")
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())
    print("\nSample rows:")
    print(df[["text", "sentiment", "brand", "platform"]].head(5).to_string(index=False))