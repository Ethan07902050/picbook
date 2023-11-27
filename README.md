# Picture Books Recommendation

## Execution
```
bash run_archiver.sh
bash run_serve.sh
```

## API
Base url: `https://140.112.29.226:8443`

### /predictions/dpr
Return a recommended picture book based on user's response.

#### Method
POST

#### Parameter
- `body`: user's query.

#### Response
- `id`: unique identifier of the picture book.
