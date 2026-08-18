from maps.apartment import APARTMENT_ASCII
print('\n'.join(APARTMENT_ASCII))
for i, row in enumerate(APARTMENT_ASCII):
    if 'F' in row:
        print('fire_row', i, 'fire_col', row.index('F'))
