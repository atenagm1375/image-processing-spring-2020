import urllib
import os
import sys
import csv
import requests

from isic_api import ISICApi


def main(offset, count, meta=True):
    api = ISICApi()

    savePath = '../../ISICArchive/'

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    imageList = api.getJson(f'image?limit={count}&offset={offset}&sort=name')

    i = count - 1
    if meta:
        print('Fetching metadata for %s images' % len(imageList))
        imageDetails = []
        for ind, image in enumerate(imageList):
            print(' ', image['name'])
            # Fetch the full image details
            try:
                imageDetail = api.getJson('image/%s' % image['_id'])
                imageDetails.append(imageDetail)
            except requests.exceptions.ConnectionError:
                imageList = api.getJson(f'image?limit={count}&offset={offset}&sort=name')
                # i = ind
                # break

        # Determine the union of all image metadata fields
        metadataFields = set(
            field
            for imageDetail in imageDetails
            for field in imageDetail['meta']['clinical'].keys()
        )
        metadataFields = ['isic_id'] + sorted(metadataFields)

        # Write the metadata to a CSV
        outputFileName = f"metadata_{offset}_{offset+i}"
        print('Writing metadata to CSV: %s' % outputFileName + '.csv')
        with open(savePath + outputFileName + '.csv', 'w') as outputStream:
            csvWriter = csv.DictWriter(outputStream, metadataFields)
            csvWriter.writeheader()
            for imageDetail in imageDetails:
                rowDict = imageDetail['meta']['clinical'].copy()
                rowDict['isic_id'] = imageDetail['name']
                csvWriter.writerow(rowDict)

    print('Downloading %s images' % len(imageList))
    imageDetails = []
    for ind, image in enumerate(imageList):
        if ind > i:
            break
        print(image['name'])
        try:
            imageFileResp = api.get('image/%s/download' % image['_id'])
            imageFileResp.raise_for_status()
            imageFileOutputPath = os.path.join(
                savePath, '%s.jpg' % image['name'])
            with open(imageFileOutputPath, 'wb') as imageFileOutputStream:
                for chunk in imageFileResp:
                    imageFileOutputStream.write(chunk)
        except requests.exceptions.ConnectionError:
            # imageList = api.getJson(
            #     f'image?limit={count-ind}&offset={offset+ind}&sort=name')
            print(ind, "FAILED.")
            break


if __name__ == "__main__":
    offset = int(sys.argv[1])
    count = 100
    if len(sys.argv) > 2:
        count = int(sys.argv[2])
    meta = True
    if len(sys.argv) > 3:
        meta = sys.argv[3]
    main(offset, count, False if meta == "False" else True)
