# Picture Books Recommendation

## Installation
```
conda env create -f environment.yml
```

## Execution
```
flask --app main run --host=0.0.0.0
```

## API
Base url: `http://140.112.29.236:5000`

### /questions
Return a random selected question.

#### Example

**Request**

`base_url`/questions

**Response**
```
{
    "question": "Are you a morning shower person or an evening shower person?"
}
```

### /search
Return a recommended picture book based on user's response.

#### Parameter
- `q`: user's query. Spaces should be replaced with '+'.

#### Response
- `id`: unique identifier of the picture book.
- `title`: title of the picture book.
- `epub_link`: link to download the epub file. 

#### Example

**Request**

`base_url`/questions?q=little+prince

**Response**
```
{
    "epub_link": "https://archive.org/download/sleepingbeautyin00perriala/sleepingbeautyin00perriala.pub",
    "id": "sleepingbeautyin00perriala",
    "title": "Sleeping beauty in the woods"
}
```