import os

# BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# BASE_PATH = os.path.join(BASE_PATH, os.pardir)

PLACEHOLDER_IMAGE_NAME = "placeholder.jpg"

# Retrieval SQL queries
TAG_QUERY = "SELECT * FROM `Tags` WHERE `Tags`.`meta.validTo` >= '2223-06-27 00:00:00.000' LIMIT 1048576"
OUTFITS_QUERY = "SELECT `Outfits`.`id` AS `id`, `Outfits`.`owner` AS `owner`, `Outfits`.`name` AS `name`, `Outfits`.`description` AS `description`, `Outfits`.`brand` AS `brand`, `Outfits`.`isPublic` AS `isPublic`, `Outfits`.`isDeleted` AS `isDeleted`, `Outfits`.`timeCreated` AS `timeCreated`, `Outfits`.`timeUpdated` AS `timeUpdated`, `Outfits`.`pricePerWeek` AS `pricePerWeek`, `Outfits`.`pricePerMonth` AS `pricePerMonth`, `Outfits`.`type` AS `type`, `Outfits`.`keywords` AS `keywords`, `Outfits`.`retailPrice` AS `retailPrice`, `Outfits`.`meta.validFrom` AS `meta.validFrom`, `Outfits`.`meta.validTo` AS `meta.validTo` FROM `Outfits` WHERE (`Outfits`.`isPublic` = TRUE AND `Outfits`.`isDeleted` = FALSE AND `Outfits`.`meta.validTo` >= '9999-01-01 00:00:00')"
PICTURES_QUERY = "SELECT * FROM Pictures WHERE (`Pictures`.`meta.validTo` >= '9999-01-01 00:00:00')"
OUTFIT_TAG_QUERY = "SELECT * FROM OutfitTags WHERE (`OutfitTags`.`meta.validTo` >= '9999-01-01 00:00:00')"
USER_ORDER_QUERY = "SELECT * FROM Orders2 WHERE `Orders2`.`meta.validTo` >= '2223-06-27 00:00:00.000' LIMIT 1048576"
USER_QUERY = "SELECT * FROM Users"
SUBSCRIPTION_RENTALS_QUERY = "SELECT * FROM `SubscriptionRentals` WHERE `SubscriptionRentals`.`meta.validTo` >= '2223-06-27 00:00:00.000' LIMIT 1048576"

# information schema queries
USER_COLUMNS_QUERY = "SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = N'Users'"

# Pandas keep columns
ORDER_KEEP_COLUMNS = ["id", "customer.id", "extras.contactEmail", "meta.validFrom", "derived.bookingTime", "shoppingCartMarker"]
OUTFITS_DF_KEEP_COLUMNS = ["id", "name", "description", "group", "timeCreated", "retailPrice", "meta.validTo"]
ORDER_KEEP_COLUMNS_USER_DATA = ["id", "customer.id", "email", "username"]
RENTALS_KEEP_COLUMNS = ["id", "order", "subscription", "outfit.id"]
PICTURES_KEEP_COLUMNS = ["id", "owner", "displayOrder"]

# CF save dirs
DATA_SAVE_PATH = "resources/data/dataframes/"

COMPRESSION_TYPE = "gzip"
COMPRESSION_EXTENSION = "gz"
# pd file names
ORDERS_PATH = f"orders_df.{COMPRESSION_EXTENSION}"
OUTFITS_PATH = f"outfits_df.{COMPRESSION_EXTENSION}"
PREDICTIONS_PATH = f"predictions_df.{COMPRESSION_EXTENSION}"
PICTURES_PATH = f"pictures_df.{COMPRESSION_EXTENSION}"
OUTFIT_FACTORS_PATH = f"outfit_factors_df.npy"
USER_FACTORS_PATH = f"user_factors_df.npy"
NEAREST_NEIGHBORS_PATH = "outfits_nearest_neighbors.pkl"

# Keep a manual record of sizes.
# Shouldn't expand too much and no sensible way of programatically find their relative sizes.
SIZE_REFERENCES = ['XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL', '4XL', '5XL']
WILDCARD_SIZES = ["Onesize", "NaN", "None", '37', '38', '41', '36', '40', '39']

# Embedding paths
PICTURES_DIR_PATH = r"C:\Users\kaborg15\PycharmProjects\FREja_API\resources\fjong_images"#"resources/fjong_images/"
LOCAL_EMBEDDINGS_PATH = "resources/picture_embeddings"
BUCKET_EMBEDDINGS_PATH = "resources/picture_embeddings/EfficientNet_V2_L_final/"

# Publishable dataset constants
DATASET_FOLDER = "resources/data/dataset/"
DATASET_IMAGES_FOLDER = "resources/data/dataset/images/"

USER_ACTIVITY_TRIPLET_COLUMNS = ["customer.id", "outfit.id", "meta.validFrom", "derived.bookingTime"]

# CSV file names
USER_ACTIVITY_TRIPLETS_CSV = "user_activity_triplets.csv"
PICTURE_TRIPLETS_CSV = "picture_triplets.csv"
OUTFITS_CSV = "outfits.csv"

USER_ACTIVITY_TRIPLETS_CSV_PATH = DATASET_FOLDER + USER_ACTIVITY_TRIPLETS_CSV
PICTURE_TRIPLETS_CSV_PATH = DATASET_FOLDER + PICTURE_TRIPLETS_CSV
OUTFITS_CSV_PATH = DATASET_FOLDER + OUTFITS_CSV

CSV_SEPARATOR = ";"