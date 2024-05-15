import os

# BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# BASE_PATH = os.path.join(BASE_PATH, os.pardir)

PLACEHOLDER_IMAGE_NAME = "placeholder.jpg"

# Retrieval SQL queries
TAG_QUERY = "SELECT * FROM `Tags` WHERE `Tags`.`meta.validTo` >= '2223-06-27 00:00:00.000' LIMIT 1048576"
#OUTFITS_QUERY = "SELECT `Outfits`.`id` AS `id`, `Outfits`.`owner` AS `owner`, `Outfits`.`name` AS `name`, `Outfits`.`description` AS `description`, `Outfits`.`brand` AS `brand`, `Outfits`.`isPublic` AS `isPublic`, `Outfits`.`isDeleted` AS `isDeleted`, `Outfits`.`timeCreated` AS `timeCreated`, `Outfits`.`timeUpdated` AS `timeUpdated`, `Outfits`.`pricePerWeek` AS `pricePerWeek`, `Outfits`.`pricePerMonth` AS `pricePerMonth`, `Outfits`.`type` AS `type`, `Outfits`.`keywords` AS `keywords`, `Outfits`.`retailPrice` AS `retailPrice`, `Outfits`.`meta.validFrom` AS `meta.validFrom`, `Outfits`.`meta.validTo` AS `meta.validTo` FROM `Outfits` WHERE (`Outfits`.`isPublic` = TRUE AND `Outfits`.`isDeleted` = FALSE AND `Outfits`.`meta.validTo` >= '9999-01-01 00:00:00')"
OUTFITS_QUERY = "SELECT `Outfits`.`id` AS `id`, `Outfits`.`barcodes` AS `barcodes`, `Outfits`.`status` AS `status`, `Outfits`.`owner` AS `owner`, `Outfits`.`name` AS `name`, `Outfits`.`brand` AS `brand`, `Outfits`.`ean` AS `ean`, `Outfits`.`primaryColor` AS `primaryColor`, `Outfits`.`isInShowroom` AS `isInShowroom`, `Outfits`.`oldPrice` AS `oldPrice`, `Outfits`.`description` AS `description`, `Outfits`.`isPublic` AS `isPublic`, `Outfits`.`isDeleted` AS `isDeleted`, `Outfits`.`stock` AS `stock`, `Outfits`.`timeCreated` AS `timeCreated`, `Outfits`.`timeUpdated` AS `timeUpdated`, `Outfits`.`rand` AS `rand`, `Outfits`.`pricePerWeek` AS `pricePerWeek`, `Outfits`.`pricePerMonth` AS `pricePerMonth`, `Outfits`.`currency` AS `currency`, `Outfits`.`type` AS `type`, `Outfits`.`keywords` AS `keywords`, `Outfits`.`retailPrice` AS `retailPrice`, `Outfits`.`measurements.length.min` AS `measurements.length.min`, `Outfits`.`measurements.length.max` AS `measurements.length.max`, `Outfits`.`measurements.length.unit` AS `measurements.length.unit`, `Outfits`.`measurements.width.min` AS `measurements.width.min`, `Outfits`.`measurements.width.max` AS `measurements.width.max`, `Outfits`.`measurements.width.unit` AS `measurements.width.unit`, `Outfits`.`measurements.depth.min` AS `measurements.depth.min`, `Outfits`.`measurements.depth.max` AS `measurements.depth.max`, `Outfits`.`measurements.depth.unit` AS `measurements.depth.unit`, `Outfits`.`measurements.weight.min` AS `measurements.weight.min`, `Outfits`.`measurements.weight.max` AS `measurements.weight.max`, `Outfits`.`measurements.weight.unit` AS `measurements.weight.unit`, `Outfits`.`measurements.shoulder.min` AS `measurements.shoulder.min`, `Outfits`.`measurements.shoulder.max` AS `measurements.shoulder.max`, `Outfits`.`measurements.shoulder.unit` AS `measurements.shoulder.unit`, `Outfits`.`measurements.sleeve.min` AS `measurements.sleeve.min`, `Outfits`.`measurements.sleeve.max` AS `measurements.sleeve.max`, `Outfits`.`measurements.sleeve.unit` AS `measurements.sleeve.unit`, `Outfits`.`measurements.bust.min` AS `measurements.bust.min`, `Outfits`.`measurements.bust.max` AS `measurements.bust.max`, `Outfits`.`measurements.bust.unit` AS `measurements.bust.unit`, `Outfits`.`measurements.waist.min` AS `measurements.waist.min`, `Outfits`.`measurements.waist.max` AS `measurements.waist.max`, `Outfits`.`measurements.waist.unit` AS `measurements.waist.unit`, `Outfits`.`measurements.hips.min` AS `measurements.hips.min`, `Outfits`.`measurements.hips.max` AS `measurements.hips.max`, `Outfits`.`measurements.hips.unit` AS `measurements.hips.unit`, `Outfits`.`measurements.rise.min` AS `measurements.rise.min`, `Outfits`.`measurements.rise.max` AS `measurements.rise.max`, `Outfits`.`measurements.rise.unit` AS `measurements.rise.unit`, `Outfits`.`measurements.inseam.min` AS `measurements.inseam.min`, `Outfits`.`measurements.inseam.max` AS `measurements.inseam.max`, `Outfits`.`measurements.inseam.unit` AS `measurements.inseam.unit`, `Outfits`.`measurements.strapLength.min` AS `measurements.strapLength.min`, `Outfits`.`measurements.strapLength.max` AS `measurements.strapLength.max`, `Outfits`.`measurements.strapLength.unit` AS `measurements.strapLength.unit`, `Outfits`.`measurements.pendantLength.min` AS `measurements.pendantLength.min`, `Outfits`.`measurements.pendantLength.max` AS `measurements.pendantLength.max`, `Outfits`.`measurements.pendantLength.unit` AS `measurements.pendantLength.unit`, `Outfits`.`measurements.pendantWidth.min` AS `measurements.pendantWidth.min`, `Outfits`.`measurements.pendantWidth.max` AS `measurements.pendantWidth.max`, `Outfits`.`measurements.pendantWidth.unit` AS `measurements.pendantWidth.unit`, `Outfits`.`accounting.purchasePrice.amount` AS `accounting.purchasePrice.amount`, `Outfits`.`accounting.purchasePrice.currency` AS `accounting.purchasePrice.currency`, `Outfits`.`accounting.purchaseDate` AS `accounting.purchaseDate`, `Outfits`.`accounting.salesPrice.amount` AS `accounting.salesPrice.amount`, `Outfits`.`accounting.salesPrice.currency` AS `accounting.salesPrice.currency`, `Outfits`.`accounting.salesDate` AS `accounting.salesDate`, `Outfits`.`accounting.estimatedLifetime` AS `accounting.estimatedLifetime`, `Outfits`.`operationalCost` AS `operationalCost`, `Outfits`.`measurements.retailSize.min` AS `measurements.retailSize.min`, `Outfits`.`measurements.retailSize.max` AS `measurements.retailSize.max`, `Outfits`.`measurements.retailSize.unit` AS `measurements.retailSize.unit`, `Outfits`.`notes.measurements` AS `notes.measurements`, `Outfits`.`notes.materials` AS `notes.materials`, `Outfits`.`meta.validFrom` AS `meta.validFrom`, `Outfits`.`meta.validTo` AS `meta.validTo`, `Outfits`.`pricing.lenderBaseFee` AS `pricing.lenderBaseFee`, `Outfits`.`pricing.lenderPerDiemFee` AS `pricing.lenderPerDiemFee`, `Outfits`.`pricing.currency` AS `pricing.currency`, `Outfits`.`pricing.subscriptionLenderBaseFee` AS `pricing.subscriptionLenderBaseFee`, `Outfits`.`group` AS `group`, `Outfits`.`pricing.subscriptionLenderPerDiemFee` AS `pricing.subscriptionLenderPerDiemFee` FROM `Outfits` LIMIT 1048576"
PICTURES_QUERY = "SELECT * FROM Pictures WHERE (`Pictures`.`meta.validTo` >= '9999-01-01 00:00:00')"
OUTFIT_TAG_QUERY = "SELECT * FROM OutfitTags WHERE (`OutfitTags`.`meta.validTo` >= '9999-01-01 00:00:00')"
USER_ORDER_QUERY = "SELECT * FROM Orders2 WHERE `Orders2`.`meta.validTo` >= '2223-06-27 00:00:00.000' LIMIT 1048576"
USER_QUERY = "SELECT * FROM Users"
SUBSCRIPTION_RENTALS_QUERY = "SELECT * FROM `SubscriptionRentals` WHERE `SubscriptionRentals`.`meta.validTo` >= '2223-06-27 00:00:00.000' LIMIT 1048576"
SPOT_RENTALS_QUERY = "SELECT * FROM `SpotRentals` WHERE `SpotRentals`.`meta.validTo` >= '2223-06-27 00:00:00.000' LIMIT 1048576"

# information schema queries
USER_COLUMNS_QUERY = "SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = N'Users'"

# Pandas keep columns
ORDER_KEEP_COLUMNS = ["id", "customer.id", "extras.contactEmail", "meta.validTo", "derived.bookingTime", "shoppingCartMarker"]
OUTFITS_DF_KEEP_COLUMNS = ["id", "name", "description", "group", "owner", "timeCreated", "retailPrice", "meta.validTo"]
ORDER_KEEP_COLUMNS_USER_DATA = ["id", "customer.id", "email", "username"]
RENTALS_KEEP_COLUMNS = ["id", "order", "subscription", "outfit.id"]
PICTURES_KEEP_COLUMNS = ["id", "owner", "displayOrder"]
SPOT_RENTALS_KEEP_COLUMNS = ["id", "order", "outfit.id", "totalPeriod.start", "totalPeriod.end", "meta.validTo"]
SPOT_RENTALS_TRIPLET_COLUMNS = ["id", "outfit.id", "totalPeriod.start", "totalPeriod.end"]
SPOT_RENTALS_RENAME_COLUMNS = {"outfit.id": "outfit_id", "totalPeriod.start": "start_date", "totalPeriod.end": "end_date"}

THIRD_CHANCE_DROP_COLUMNS = [False, "LINK", "COMMENT", "SHOPIFY", "LASTET OPP", "LOCATION", "OK TO SELL?", "STATUS", "LOCATED", "BARCODE"]

# CF save dirs
DATA_SAVE_PATH = "resources/data/dataframes/"

COMPRESSION_TYPE = "gzip"
COMPRESSION_EXTENSION = "gz"
# pd file names
ORDERS_PATH = f"orders_df.{COMPRESSION_EXTENSION}"
OUTFITS_PATH = f"outfits_df.{COMPRESSION_EXTENSION}"
OUTFITS_FULL_PATH = f"outfits_full_df.{COMPRESSION_EXTENSION}"
PREDICTIONS_PATH = f"predictions_df.{COMPRESSION_EXTENSION}"
PICTURES_PATH = f"pictures_df.{COMPRESSION_EXTENSION}"
SPOT_RENTALS_PATH = f"spot_rentals_df.{COMPRESSION_EXTENSION}"
OUTFIT_FACTORS_PATH = f"outfit_factors_df.npy"
USER_FACTORS_PATH = f"user_factors_df.npy"
NEAREST_NEIGHBORS_PATH = "outfits_nearest_neighbors.pkl"

# Other data file paths
THIRD_CHANCE_RAW_PATH = "resources/data/third_chance.xlsx"

# Keep a manual record of sizes.
# Shouldn't expand too much and no sensible way of programatically find their relative sizes.
SIZE_REFERENCES = ['XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL', '4XL', '5XL']
WILDCARD_SIZES = ["Onesize", "NaN", "None", '37', '38', '41', '36', '40', '39']

# Embedding paths
EMBEDDING_MODEL_NAME = "EfficientNet_V2_L_final"
PICTURES_DIR_PATH = r"C:\Users\kaborg15\PycharmProjects\FREja_API\resources\fjong_images"#"resources/fjong_images/"
LOCAL_EMBEDDINGS_FOLDER_PATH = "resources/data/dataset/embeddings/"
COMPUTED_EMBEDDINGS_PATH = f"resources/data/dataset/embeddings/{EMBEDDING_MODEL_NAME}/"
EMBEDDING_MODEL_PICKLE_PATH = f"{LOCAL_EMBEDDINGS_FOLDER_PATH}{EMBEDDING_MODEL_NAME}.pkl"
EMBEDDING_MODEL_DICT_PICKLE_PATH = f"{LOCAL_EMBEDDINGS_FOLDER_PATH}{EMBEDDING_MODEL_NAME}_dict.pkl"

# Publishable dataset constants
DATASET_FOLDER = "resources/data/dataset/"
DATASET_IMAGES_FOLDER = "resources/data/dataset/images/"
DATASET_IMAGE_EMBEDDINGS_FOLDER = "resources/data/dataset/embeddings/"

USER_ACTIVITY_TRIPLET_COLUMNS = ["customer.id", "outfit.id", "meta.validFrom", "derived.bookingTime"]

# CSV file names
USER_ACTIVITY_TRIPLETS_CSV = "user_activity_triplets.csv"
SPOT_RENTALS_CSV = "spot_rentals.csv"
PICTURE_TRIPLETS_CSV = "picture_triplets.csv"
OUTFITS_CSV = "outfits.csv"
THIRD_CHANCE_CSV = "third_chance.csv"

USER_ACTIVITY_TRIPLETS_CSV_PATH = DATASET_FOLDER + USER_ACTIVITY_TRIPLETS_CSV
SPOT_RENTALS_CSV_PATH = DATASET_FOLDER + SPOT_RENTALS_CSV
PICTURE_TRIPLETS_CSV_PATH = DATASET_FOLDER + PICTURE_TRIPLETS_CSV
OUTFITS_CSV_PATH = DATASET_FOLDER + OUTFITS_CSV
THIRD_CHANCE_CSV_PATH = DATASET_FOLDER + THIRD_CHANCE_CSV

CSV_SEPARATOR = ";"